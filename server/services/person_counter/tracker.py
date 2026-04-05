"""ByteTrack object tracker module."""

from __future__ import annotations

import supervision as sv
from loguru import logger

from config import settings


class PersonTracker:
    """ByteTrack-based person tracker with frame skip management."""

    def __init__(self) -> None:
        self.byte_track = sv.ByteTrack()
        self.frame_count: int = 0
        logger.info(
            "PersonTracker initialized: frame_skip={}",
            settings.FRAME_SKIP,
        )

    def should_detect(self) -> bool:
        """Return True if YOLO detection should run on the current frame."""
        return self.frame_count % settings.FRAME_SKIP == 0

    def increment_frame(self) -> None:
        """Advance the internal frame counter by one."""
        self.frame_count += 1

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update ByteTrack with new detections and return tracked detections.

        Args:
            detections: sv.Detections from the detector (or empty for skipped frames).

        Returns:
            sv.Detections with tracker_id assigned.
        """
        tracked = self.byte_track.update_with_detections(detections)
        return tracked

    def reset(self) -> None:
        """Reset the tracker state."""
        self.byte_track.reset()
        self.frame_count = 0
        logger.info("PersonTracker reset")
