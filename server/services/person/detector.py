"""YOLOv8 person detector — ROI 지원, CUDA 가속."""

import logging

import numpy as np
import supervision as sv
from ultralytics import YOLO

from server.config import settings

logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLOv8 기반 사람 감지기. ROI 크롭 + 원본 좌표 복원."""

    def __init__(self) -> None:
        model_path = settings.PERSON_YOLO_MODEL
        self._model = YOLO(model_path)
        self._confidence = settings.PERSON_CONFIDENCE_THRESHOLD
        self._model.to("cuda:0")
        logger.info(
            "PersonDetector initialized: model=%s confidence=%.2f",
            model_path,
            self._confidence,
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """ROI 영역을 크롭해 YOLO 추론 후 원본 좌표로 변환해 반환."""
        rx = settings.PERSON_ROI_X
        ry = settings.PERSON_ROI_Y
        rw = settings.PERSON_ROI_W
        rh = settings.PERSON_ROI_H

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
