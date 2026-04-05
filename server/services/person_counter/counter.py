"""Line crossing counter logic module."""

from __future__ import annotations

from dataclasses import dataclass

import supervision as sv
from loguru import logger

from config import settings


@dataclass
class CountEvent:
    """Represents a single line-crossing event for backend integration."""

    person_id: int
    direction: str  # "IN" or "OUT"
    count_change: int  # +1 or -1
    confidence: float
    bbox: tuple[float, float, float, float]


class LineCrossCounter:
    """Counts persons crossing a configurable line using center-point tracking."""

    def __init__(self) -> None:
        self.line_start: tuple[int, int] = (
            settings.LINE_START_X,
            settings.LINE_START_Y,
        )
        self.line_end: tuple[int, int] = (settings.LINE_END_X, settings.LINE_END_Y)

        # Line equation coefficients: a*x + b*y + c = 0
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        self._a = y2 - y1
        self._b = x1 - x2
        self._c = x2 * y1 - x1 * y2

        self.counted_ids: set[int] = set()
        self.last_positions: dict[int, float] = {}  # tracker_id -> signed distance
        self.last_direction: dict[int, str] = {}

        self.in_count: int = 0
        self.out_count: int = 0
        self._current_count: int = 0

        logger.info(
            "LineCrossCounter initialized: line=({},{})→({},{}), direction={}",
            x1,
            y1,
            x2,
            y2,
            settings.LINE_DIRECTION,
        )

    def _signed_distance(self, cx: float, cy: float) -> float:
        """Compute the signed distance from point (cx, cy) to the counting line.

        Positive values mean one side, negative the other.
        """
        return self._a * cx + self._b * cy + self._c

    def update(self, detections: sv.Detections) -> list[CountEvent]:
        """Process tracked detections and return line-crossing events.

        Args:
            detections: sv.Detections with tracker_id assigned by ByteTrack.

        Returns:
            List of CountEvent for each new crossing detected this frame.
        """
        events: list[CountEvent] = []

        if detections.tracker_id is None or len(detections) == 0:
            return events

        for i in range(len(detections)):
            tracker_id = int(detections.tracker_id[i])
            x1, y1, x2, y2 = detections.xyxy[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            current_dist = self._signed_distance(cx, cy)

            if tracker_id in self.last_positions:
                prev_dist = self.last_positions[tracker_id]

                # Check for sign change (line crossing)
                if prev_dist * current_dist < 0 and tracker_id not in self.counted_ids:
                    confidence = (
                        float(detections.confidence[i])
                        if detections.confidence is not None
                        else 0.0
                    )
                    bbox = (float(x1), float(y1), float(x2), float(y2))

                    # Determine direction: negative→positive = IN, positive→negative = OUT
                    if prev_dist < 0 and current_dist > 0:
                        direction = "IN"
                        count_change = 1
                        self.in_count += 1
                    else:
                        direction = "OUT"
                        count_change = -1
                        self.out_count += 1

                    self._current_count += count_change
                    self.counted_ids.add(tracker_id)
                    self.last_direction[tracker_id] = direction

                    event = CountEvent(
                        person_id=tracker_id,
                        direction=direction,
                        count_change=count_change,
                        confidence=confidence,
                        bbox=bbox,
                    )
                    events.append(event)
                    logger.info(
                        "Line crossed: person_id={}, direction={}, current_count={}",
                        tracker_id,
                        direction,
                        self._current_count,
                    )

            self.last_positions[tracker_id] = current_dist

        return events

    @property
    def current_count(self) -> int:
        """Net count of people (IN - OUT)."""
        return self._current_count
