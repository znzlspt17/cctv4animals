"""People Counter 서비스 — FastAPI 백그라운드 스레드로 실행.

lifespan에서 CameraManager.start_all(configs, repo) 를 호출한다.
각 카메라마다 PersonCounterService 인스턴스 1개가 독립 스레드로 동작한다.
"""

import logging
import threading
import time
from typing import Any

import cv2
import supervision as sv

from server.services.common.line_tracker import LineCrossTracker
from server.services.common.result_publisher import publish_event
from server.services.person.camera_config import CameraConfig
from server.services.person.detector import PersonDetector

logger = logging.getLogger(__name__)


class PersonCounterService:
    """단일 카메라를 전담하는 사람 카운팅 백그라운드 서비스."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()   # set → 카메라 해제 요청
        self._paused_ack = threading.Event()    # set → 카메라 실제로 해제됨

        # 공유 상태 (읽기는 GIL 보호로 충분)
        self._current_count: int = 0
        self._in_count: int = 0
        self._out_count: int = 0
        self._running: bool = False
        self._error: str | None = None

    # ──────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────

    @property
    def camera_id(self) -> str:
        return self._config.camera_id

    @property
    def config(self) -> CameraConfig:
        return self._config

    def start(self, repo: Any) -> None:
        """백그라운드 카운팅 스레드를 시작한다."""
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(repo,),
            daemon=True,
            name=f"PersonCounter-{self._config.camera_id}",
        )
        self._thread.start()
        self._running = True
        logger.info(
            "PersonCounterService started (camera_id=%s, source=%s)",
            self._config.camera_id,
            self._config.video_source,
        )

    def stop(self) -> None:
        """루프에 종료 신호를 보내고 스레드가 끝날 때까지 기다린다."""
        if not self._running:
            return
        logger.info("PersonCounterService stopping…")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._running = False
        logger.info("PersonCounterService stopped")

    # ──────────────────────────────────────────────
    # 상태 조회
    # ──────────────────────────────────────────────

    @property
    def current_count(self) -> int:
        return self._current_count

    @property
    def in_count(self) -> int:
        return self._in_count

    @property
    def out_count(self) -> int:
        return self._out_count

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused_ack.is_set()

    def pause_camera(self, timeout: float = 5.0) -> bool:
        """카메라를 해제하고 루프를 일시정지한다. 웹캠 공용 사용 시 호출."""
        if not self._running:
            return True
        self._paused_ack.clear()
        self._pause_event.set()
        acquired = self._paused_ack.wait(timeout=timeout)
        if acquired:
            logger.info("PersonCounter: camera paused (released)")
        else:
            logger.warning("PersonCounter: pause timeout — camera may still be held")
        return acquired

    def resume_camera(self) -> None:
        """일시정지된 카메라 루프를 재개한다."""
        self._pause_event.clear()
        logger.info("PersonCounter: camera resume requested")

    @property
    def error(self) -> str | None:
        return self._error

    def get_stats(self) -> dict[str, Any]:
        return {
            "camera_id": self._config.camera_id,
            "label": self._config.label,
            "video_source": self._config.video_source,
            "current_count": self._current_count,
            "in_count": self._in_count,
            "out_count": self._out_count,
            "running": self._running,
            "paused": self.is_paused,
            "error": self._error,
        }

    # ──────────────────────────────────────────────
    # 내부: 카운트 복원
    # ──────────────────────────────────────────────

    def _restore_count(self, repo: Any) -> int:
        """DB에서 마지막 current_count를 읽어 복원한다."""
        try:
            count = repo.tracking_event.get_current_count(self._config.camera_id)
            logger.info(
                "PersonCounter[%s]: restored count=%d",
                self._config.camera_id, count,
            )
            return count
        except Exception as e:
            logger.warning(
                "PersonCounter[%s]: count restore failed (%s) — starting from 0",
                self._config.camera_id, e,
            )
            return 0

    # ──────────────────────────────────────────────
    # 내부: 메인 루프
    # ──────────────────────────────────────────────

    def _run_loop(self, repo: Any) -> None:
        """카메라 캡처 → 감지 → 추적 → 라인 크로싱 메인 루프."""
        try:
            self._loop_body(repo)
        except Exception as e:
            self._error = str(e)
            logger.exception("PersonCounterService loop crashed: %s", e)
        finally:
            self._running = False

    def _loop_body(self, repo: Any) -> None:
        # 카운트 복원
        self._current_count = self._restore_count(repo)
        cfg = self._config

        # 모듈 초기화
        roi = (cfg.roi_x, cfg.roi_y, cfg.roi_w, cfg.roi_h) if cfg.roi_w and cfg.roi_h else None
        detector = PersonDetector(roi=roi, confidence=cfg.confidence_threshold)
        byte_tracker = sv.ByteTrack()
        frame_count = 0
        line_tracker = LineCrossTracker(
            x1=cfg.line_start_x,
            y1=cfg.line_start_y,
            x2=cfg.line_end_x,
            y2=cfg.line_end_y,
        )

        # 비디오 소스 열기
        source = cfg.capture_source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {cfg.video_source}")
        logger.info("PersonCounter[%s]: video source opened (%s)", cfg.camera_id, cfg.video_source)

        try:
            while not self._stop_event.is_set():
                # ── 일시정지 처리 ──────────────────────────────
                if self._pause_event.is_set():
                    cap.release()
                    self._paused_ack.set()
                    logger.info("PersonCounter[%s]: camera released, waiting for resume…", cfg.camera_id)
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.1)
                    if self._stop_event.is_set():
                        return
                    self._paused_ack.clear()
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot re-open video source: {cfg.video_source}")
                    logger.info("PersonCounter[%s]: camera re-opened after resume", cfg.camera_id)
                # ────────────────────────────────────────────────

                ret, frame = cap.read()
                if not ret:
                    logger.warning("PersonCounter[%s]: frame read failed, retrying…", cfg.camera_id)
                    time.sleep(0.1)
                    continue

                frame_count += 1

                # 감지 또는 빈 Detections (프레임 스킵)
                if frame_count % cfg.frame_skip == 0:
                    detections = detector.detect(frame)
                else:
                    detections = sv.Detections.empty()

                tracked = byte_tracker.update_with_detections(detections)

                if tracked.tracker_id is None or len(tracked) == 0:
                    continue

                # 라인 크로싱 감지
                centers = [
                    (
                        (float(x1) + float(x2)) / 2.0,
                        (float(y1) + float(y2)) / 2.0,
                    )
                    for x1, y1, x2, y2 in tracked.xyxy
                ]
                track_ids = [int(tid) for tid in tracked.tracker_id]
                raw_events = line_tracker.update(centers, track_ids)

                for track_id, direction in raw_events:
                    idx = track_ids.index(track_id)
                    conf = float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0
                    x1, y1, x2, y2 = tracked.xyxy[idx]
                    count_change = 1 if direction == "IN" else -1

                    # 카운트 갱신
                    self._current_count += count_change
                    if direction == "IN":
                        self._in_count += 1
                    else:
                        self._out_count += 1

                    logger.info(
                        "PersonCounter[%s]: line crossed track_id=%d direction=%s count=%d",
                        cfg.camera_id, track_id, direction, self._current_count,
                    )

                    snap_path = ""

                    # DB 저장
                    try:
                        repo.tracking_event.save(
                            camera_id=cfg.camera_id,
                            tracker_id=track_id,
                            direction=direction,
                            count_change=count_change,
                            current_count=self._current_count,
                            confidence=conf,
                            bbox_x=int(x1),
                            bbox_y=int(y1),
                            bbox_w=int(x2 - x1),
                            bbox_h=int(y2 - y1),
                            snapshot_path=snap_path,
                        )
                    except Exception as e:
                        logger.error("PersonCounter[%s]: DB save failed: %s", cfg.camera_id, e)

                    # 외부 서버 전송
                    publish_event(
                        event_type="person_crossing",
                        camera_id=cfg.camera_id,
                        payload={
                            "track_id": track_id,
                            "direction": direction,
                            "count_change": count_change,
                            "current_count": self._current_count,
                            "confidence": round(conf, 4),
                            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                            "snapshot_path": snap_path,
                        },
                    )

        finally:
            cap.release()
            logger.info("PersonCounter[%s]: camera released", cfg.camera_id)
