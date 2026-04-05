"""YOLOv8 person detector module."""

from __future__ import annotations

import numpy as np
import supervision as sv
from loguru import logger
from ultralytics import YOLO

from config import settings


class PersonDetector:
    """YOLOv8-based person detector with ROI support and CUDA acceleration."""

    def __init__(self) -> None:
        self.model = YOLO(settings.YOLO_MODEL)
        self.device = "cuda:0"
        self.confidence = settings.CONFIDENCE_THRESHOLD
        self.model.to(self.device)
        logger.info(
            "PersonDetector initialized: model={}, device={}, confidence={}",
            settings.YOLO_MODEL,
            self.device,
            self.confidence,
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run YOLO inference on the ROI crop and return detections in original coords.

        Args:
            frame: Full-resolution BGR frame from the camera.

        Returns:
            sv.Detections with bounding boxes mapped back to the original frame.
        """
        roi_x = settings.ROI_X
        roi_y = settings.ROI_Y
        roi_w = settings.ROI_W
        roi_h = settings.ROI_H

        # Crop the ROI region
        cropped = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        results = self.model(
            cropped,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        detections = sv.Detections.from_ultralytics(result)

        # Filter person class only (class_id == 0)
        if len(detections) > 0:
            person_mask = detections.class_id == 0
            detections = detections[person_mask]

        # Translate bounding boxes back to original frame coordinates
        if len(detections) > 0:
            detections.xyxy[:, 0] += roi_x  # x1
            detections.xyxy[:, 1] += roi_y  # y1
            detections.xyxy[:, 2] += roi_x  # x2
            detections.xyxy[:, 3] += roi_y  # y2

        return detections
