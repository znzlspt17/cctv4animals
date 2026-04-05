"""Snapshot and recording management module."""

from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np
from loguru import logger

from counter import CountEvent


class SnapshotManager:
    """Saves full-frame and cropped snapshots on counting events."""

    def __init__(self, base_dir: str = "snapshots") -> None:
        self.base_dir = base_dir
        logger.info("SnapshotManager initialized: base_dir={}", self.base_dir)

    def save(self, frame: np.ndarray, event: CountEvent, camera_id: str) -> str:
        """Save a full-frame JPEG and a bbox crop for a counting event.

        Args:
            frame: The full BGR frame at the time of the event.
            event: The CountEvent that triggered the snapshot.
            camera_id: Camera identifier for directory organisation.

        Returns:
            The path to the saved full-frame snapshot.
        """
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")

        dir_path = os.path.join(self.base_dir, camera_id, date_str)
        os.makedirs(dir_path, exist_ok=True)

        # Full frame
        filename = f"{event.person_id}_{timestamp_str}.jpg"
        full_path = os.path.join(dir_path, filename)
        cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Cropped bbox
        x1, y1, x2, y2 = event.bbox
        h, w = frame.shape[:2]
        cx1 = max(0, int(x1))
        cy1 = max(0, int(y1))
        cx2 = min(w, int(x2))
        cy2 = min(h, int(y2))

        if cx2 > cx1 and cy2 > cy1:
            crop = frame[cy1:cy2, cx1:cx2]
            crop_filename = f"{event.person_id}_{timestamp_str}_crop.jpg"
            crop_path = os.path.join(dir_path, crop_filename)
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

        logger.debug("Snapshot saved: {}", full_path)
        return full_path
