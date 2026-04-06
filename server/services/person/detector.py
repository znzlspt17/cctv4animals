"""YOLOv8 person detector — ROI 지원, CUDA 가속."""

import logging

import numpy as np
import supervision as sv
from ultralytics import YOLO

from server.config import settings

logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLOv8 기반 사람 감지기. ROI 크롭 + 원본 좌표 복원.

    Args:
        roi: (x, y, w, h) 튜플. None 이면 settings.PERSON_ROI_* 전역값 사용.
        confidence: 탐지 임계값. None 이면 settings.PERSON_CONFIDENCE_THRESHOLD 사용.
    """

    def __init__(
        self,
        roi: tuple[int, int, int, int] | None = None,
        confidence: float | None = None,
    ) -> None:
        model_path = settings.PERSON_YOLO_MODEL
        self._model = YOLO(model_path)
        self._confidence = confidence if confidence is not None else settings.PERSON_CONFIDENCE_THRESHOLD
        if roi is not None:
            self._roi = roi
        else:
            self._roi = (
                settings.PERSON_ROI_X,
                settings.PERSON_ROI_Y,
                settings.PERSON_ROI_W,
                settings.PERSON_ROI_H,
            )
        self._model.to("cuda:0")
        logger.info(
            "PersonDetector initialized: model=%s confidence=%.2f roi=%s",
            model_path,
            self._confidence,
            self._roi,
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """ROI 영역을 크롭해 YOLO 추론 후 원본 좌표로 변환해 반환."""
        rx, ry, rw, rh = self._roi

        cropped = frame[ry: ry + rh, rx: rx + rw]
        results = self._model(cropped, conf=self._confidence, device="cuda:0", verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        # 사람 클래스(class_id=0)만 필터
        if len(detections) > 0:
            detections = detections[detections.class_id == 0]

        # bbox를 원본 좌표로 복원
        if len(detections) > 0:
            detections.xyxy[:, 0] += rx
            detections.xyxy[:, 1] += ry
            detections.xyxy[:, 2] += rx
            detections.xyxy[:, 3] += ry

        return detections
