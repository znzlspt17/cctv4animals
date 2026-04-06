"""People Counter 서비스 — FastAPI 백그라운드 스레드로 실행.

lifespan에서 person_service.start(repo) 호출 → 카메라 루프 시작.
FastAPI 종료 시 person_service.stop() → 루프 안전 종료.
"""

import logging
import threading
import time

import cv2
import supervision as sv

from server.config import settings
from server.services.common.line_tracker import LineCrossTracker
from server.services.common.result_publisher import publish_event
from server.services.person.detector import PersonDetector

logger = logging.getLogger(__name__)


class PersonCounterService:
    """카메라에서 프레임을 읽어 사람 카운팅을 수행하는 백그라운드 서비스."""

    def __init__(self) -> None:
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

    def start(self, repo) -> None:
        """백그라운드 카운팅 스레드를 시작한다."""
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(repo,),
            daemon=True,
            name="PersonCounterLoop",
        )
        self._thread.start()
        self._running = True
        logger.info("PersonCounterService started (camera=%s)", settings.PERSON_VIDEO_SOURCE)

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

    def get_stats(self) -> dict:
        return {
            "current_count": self._current_count,
            "in_count": self._in_count,
            "out_count": self._out_count,
            "camera_id": settings.PERSON_CAMERA_ID,
            "running": self._running,
            "error": self._error,
        }

    # ──────────────────────────────────────────────
    # 내부: 카운트 복원
    # ──────────────────────────────────────────────

    def _restore_count(self, repo) -> int:
        """DB에서 마지막 current_count를 읽어 복원한다."""
        try:
            count = repo.tracking_event.get_current_count(settings.PERSON_CAMERA_ID)
            logger.info("PersonCounter: restored count=%d (camera=%s)", count, settings.PERSON_CAMERA_ID)
            return count
        except Exception as e:
            logger.warning("PersonCounter: count restore failed (%s) — starting from 0", e)
            return 0

    # ──────────────────────────────────────────────
    # 내부: 메인 루프
    # ──────────────────────────────────────────────

    def _run_loop(self, repo) -> None:
        """카메라 캡처 → 감지 → 추적 → 라인 크로싱 메인 루프."""
        try:
            self._loop_body(repo)
        except Exception as e:
            self._error = str(e)
            logger.exception("PersonCounterService loop crashed: %s", e)
        finally:
            self._running = False

    def _loop_body(self, repo) -> None:
        # 카운트 복원
        self._current_count = self._restore_count(repo)

        # 모듈 초기화
        detector = PersonDetector()
        byte_tracker = sv.ByteTrack()
        frame_count = 0
        line_tracker = LineCrossTracker(
            x1=settings.PERSON_LINE_START_X,
            y1=settings.PERSON_LINE_START_Y,
            x2=settings.PERSON_LINE_END_X,
            y2=settings.PERSON_LINE_END_Y,
        )

        # 비디오 소스 열기
        source_raw = settings.PERSON_VIDEO_SOURCE
        source = int(source_raw) if source_raw.isdigit() else source_raw
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source_raw}")
        logger.info("PersonCounter: video source opened (%s)", source_raw)

        try:
            while not self._stop_event.is_set():
                # ── 일시정지 처리 ──────────────────────────────
                if self._pause_event.is_set():
                    cap.release()
                    self._paused_ack.set()
                    logger.info("PersonCounter: camera released, waiting for resume…")
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.1)
                    if self._stop_event.is_set():
                        return
                    self._paused_ack.clear()
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot re-open video source: {source_raw}")
                    logger.info("PersonCounter: camera re-opened after resume")
                # ────────────────────────────────────────────────

                ret, frame = cap.read()
                if not ret:
                    logger.warning("PersonCounter: frame read failed, retrying…")
                    time.sleep(0.1)
                    continue

                frame_count += 1

                # 감지 또는 빈 Detections (프레임 스킵)
                if frame_count % settings.PERSON_FRAME_SKIP == 0:
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
                        "PersonCounter: line crossed track_id=%d direction=%s count=%d",
                        track_id, direction, self._current_count,
                    )

                    snap_path = ""

                    # DB 저장
                    try:
                        repo.tracking_event.save(
                            camera_id=settings.PERSON_CAMERA_ID,
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
                        logger.error("PersonCounter: DB save failed: %s", e)

                    # 외부 서버 전송
                    publish_event(
                        event_type="person_crossing",
                        camera_id=settings.PERSON_CAMERA_ID,
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
            logger.info("PersonCounter: camera released")


# 싱글톤
person_counter_service = PersonCounterService()
