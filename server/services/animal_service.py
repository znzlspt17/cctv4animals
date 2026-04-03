import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# 사전학습 모델 파일 경로
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best-yolo26m-hpo.pt"

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


class AnimalService:
    """Yolo26m-hpo 기반 동물 탐지 서비스 — 싱글톤으로 사용."""

    def __init__(self):
        self._model = None
        self._loaded: bool = False

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
    # 추론
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
