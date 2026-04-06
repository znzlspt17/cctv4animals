import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from server.config import settings
from server.services.common.line_tracker import LineCrossTracker

logger = logging.getLogger(__name__)

# 사전학습 모델 파일 경로
MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "best-4animals-yolo26m-hpo.pt"

# 탐지 대상 클래스명 (빈 set이면 모델의 모든 클래스 반환)
ANIMAL_CLASSES: set[str] = set()


@dataclass
class Detection:
    """단일 객체 탐지 결과."""
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] — 픽셀 좌표


@dataclass
class AnimalDetectionResult:
    """단일 프레임의 동물 탐지 결과."""
    detections: list[Detection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.detections)

    def __repr__(self) -> str:
        return f"AnimalDetectionResult(detections={len(self.detections)})"

    @property
    def has_animal(self) -> bool:
        return len(self.detections) > 0


@dataclass
class AnimalCrossEvent:
    """라인 크로싱 감지 이벤트."""
    track_id: int
    class_name: str
    direction: str        # "IN" or "OUT"
    confidence: float
    bbox: list[float]     # [x1, y1, x2, y2]


def _parse_results(results, conf_threshold: float) -> list[Detection]:
    """ultralytics Results 객체 → Detection 리스트 변환."""
    detections: list[Detection] = []
    for r in results:
        if r.boxes is None:
            continue
        names = r.names  # {int: str}
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, str(cls_id))
            if ANIMAL_CLASSES and class_name not in ANIMAL_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                )
            )
    return detections


class _AnimalLineCrossTracker:
    """동물 객체의 라인 크로싱을 추적하는 내부 클래스.

    공통 LineCrossTracker를 감싸서 AnimalCrossEvent를 생성한다.
    YOLO tracking ID(int)를 키로 이전 위치를 기억한다.
    """

    def __init__(self, x1=None, y1=None, x2=None, y2=None) -> None:
        x1 = x1 if x1 is not None else settings.ANIMAL_LINE_START_X
        y1 = y1 if y1 is not None else settings.ANIMAL_LINE_START_Y
        x2 = x2 if x2 is not None else settings.ANIMAL_LINE_END_X
        y2 = y2 if y2 is not None else settings.ANIMAL_LINE_END_Y
        self._tracker = LineCrossTracker(x1, y1, x2, y2)
        # track_id → Detection (이벤트 생성 시 클래스/신뢰도/bbox 조회용)
        self._det_map: dict[int, Detection] = {}
        logger.info(
            "AnimalLineCrossTracker initialized: line=(%d,%d)→(%d,%d)",
            x1, y1, x2, y2,
        )

    def update(self, detections: list[Detection], track_ids: list[int]) -> list[AnimalCrossEvent]:
        """매 프레임마다 호출. 라인을 교차한 객체의 이벤트 목록을 반환."""
        centers = [
            ((d.bbox[0] + d.bbox[2]) / 2.0, (d.bbox[1] + d.bbox[3]) / 2.0)
            for d in detections
        ]
        self._det_map = {tid: det for tid, det in zip(track_ids, detections)}
        raw_events = self._tracker.update(centers, track_ids)

        events: list[AnimalCrossEvent] = []
        for track_id, direction in raw_events:
            det = self._det_map.get(track_id)
            if det is None:
                continue
            event = AnimalCrossEvent(
                track_id=track_id,
                class_name=det.class_name,
                direction=direction,
                confidence=det.confidence,
                bbox=det.bbox,
            )
            events.append(event)
            logger.info(
                "Animal line crossed: track_id=%d class=%s direction=%s",
                track_id, det.class_name, direction,
            )
        return events

    def reset(self) -> None:
        self._tracker.reset()
        self._det_map.clear()


class AnimalService:
    """Yolo26m-hpo 기반 동물 탐지 서비스 — 싱글톤으로 사용."""

    def __init__(self):
        self._model = None
        self._loaded: bool = False
        self._line_tracker = _AnimalLineCrossTracker()

    # ──────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────
    def load(self) -> None:
        """모델 파일을 로드한다. lifespan 또는 첫 호출 시 실행."""
        if self._loaded:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {MODEL_PATH}\n"
                "models/ 디렉토리에 best-yolo26m-hpo.pt 파일을 배치하세요."
            )

        from ultralytics import YOLO  # lazy import — ultralytics 없어도 서버 기동 가능

        self._model = YOLO(str(MODEL_PATH))
        self._loaded = True
        logger.info("AnimalService: model loaded from %s", MODEL_PATH)

    # ──────────────────────────────────────────────
    # 추론 (단순 탐지 — 라인 무관)
    # ──────────────────────────────────────────────
    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.4,
    ) -> AnimalDetectionResult:
        """BGR 프레임에서 동물을 탐지하고 결과를 반환한다.

        Args:
            frame: OpenCV BGR 이미지 (H x W x 3, uint8)
            conf_threshold: 이 값 이상의 confidence만 반환 (0~1)

        Returns:
            AnimalDetectionResult
        """
        if not self._loaded:
            self.load()

        results = self._model(frame, conf=conf_threshold, verbose=False)
        detections = _parse_results(results, conf_threshold)
        return AnimalDetectionResult(detections=detections)

    # ──────────────────────────────────────────────
    # 추론 + 라인 크로싱 감지
    # ──────────────────────────────────────────────
    def detect_with_crossing(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.4,
    ) -> tuple[AnimalDetectionResult, list[AnimalCrossEvent]]:
        """BGR 프레임에서 동물을 탐지하고, 라인을 넘은 객체의 이벤트를 반환한다.

        YOLO track(persist=True)으로 연속 tracking ID를 부여한 뒤,
        이전 프레임 대비 부호 거리 변화로 라인 교차를 감지한다.

        Args:
            frame: OpenCV BGR 이미지 (H x W x 3, uint8)
            conf_threshold: confidence 하한

        Returns:
            (AnimalDetectionResult, list[AnimalCrossEvent])
            — 라인을 넘지 않은 프레임이면 이벤트 리스트는 빈 리스트.
        """
        if not self._loaded:
            self.load()

        # persist=True → 동일 객체에 일관된 track ID 부여
        results = self._model.track(frame, conf=conf_threshold, persist=True, verbose=False)

        detections: list[Detection] = []
        track_ids: list[int] = []

        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, str(cls_id))
                if ANIMAL_CLASSES and class_name not in ANIMAL_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # tracking ID가 없는 박스는 스킵 (첫 프레임 등)
                if box.id is None:
                    continue
                track_id = int(box.id[0])

                detections.append(Detection(class_name=class_name, confidence=conf, bbox=[x1, y1, x2, y2]))
                track_ids.append(track_id)

        events = self._line_tracker.update(detections, track_ids)
        return AnimalDetectionResult(detections=detections), events

    def reset_tracker(self) -> None:
        """카메라 전환 등으로 추적 상태를 초기화할 때 호출."""
        self._line_tracker.reset()
        logger.info("AnimalService: line tracker reset")

    def reconfigure_line(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """라인 크로싱 기준선을 동적으로 재설정한다."""
        self._line_tracker = _AnimalLineCrossTracker(x1, y1, x2, y2)
        logger.info("AnimalService: line reconfigured to (%d,%d)→(%d,%d)", x1, y1, x2, y2)

    # ──────────────────────────────────────────────
    # 상태
    # ──────────────────────────────────────────────
    def is_ready(self) -> bool:
        return self._loaded

    def warmup(self) -> None:
        """더미 이미지로 모델을 프리로드해 첫 추론 지연을 없앤다."""
        if not self._loaded:
            self.load()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, conf=0.1, verbose=False)
        logger.info("AnimalService: warmup complete")


# 싱글톤 인스턴스
animal_service = AnimalService()
