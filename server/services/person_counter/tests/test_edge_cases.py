"""Edge case tests (Step 5.3)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_detections(xyxy_list, tracker_ids, confidences=None):
    """Build a mock detections object."""
    det = MagicMock()
    det.xyxy = (
        np.array(xyxy_list, dtype=np.float64)
        if xyxy_list
        else np.empty((0, 4), dtype=np.float64)
    )
    det.tracker_id = (
        np.array(tracker_ids, dtype=np.int64) if tracker_ids is not None else None
    )
    if confidences is not None:
        det.confidence = np.array(confidences, dtype=np.float64)
    elif xyxy_list:
        det.confidence = np.ones(len(xyxy_list), dtype=np.float64) * 0.9
    else:
        det.confidence = np.empty(0, dtype=np.float64)
    det.__len__ = lambda self: len(xyxy_list) if xyxy_list else 0
    return det


@pytest.fixture
def counter():
    """LineCrossCounter with horizontal line at y=240."""
    with patch("counter.settings") as mock_s:
        mock_s.LINE_START_X = 0
        mock_s.LINE_START_Y = 240
        mock_s.LINE_END_X = 640
        mock_s.LINE_END_Y = 240
        mock_s.LINE_DIRECTION = "horizontal"
        from counter import LineCrossCounter

        return LineCrossCounter()


class TestEmptyDetections:
    def test_empty_detections(self, counter):
        """Empty detections → no events, no errors."""
        det = _make_detections([], None)
        events = counter.update(det)
        assert events == []

    def test_zero_length_detections(self, counter):
        """Detections with tracker_id=None → no events."""
        det = _make_detections([[100, 100, 200, 200]], None)
        events = counter.update(det)
        assert events == []


class TestNoTrackerId:
    def test_no_tracker_id(self, counter):
        """tracker_id=None → gracefully returns empty list."""
        det = MagicMock()
        det.tracker_id = None
        det.__len__ = lambda self: 1
        events = counter.update(det)
        assert events == []


class TestManyDetections:
    def test_many_detections_crossing(self, counter):
        """10+ simultaneous persons crossing → all counted correctly."""
        n = 12
        # Frame 1: all persons below the line
        xyxy_below = [[50 * i, 280, 50 * i + 40, 320] for i in range(n)]
        ids = list(range(1, n + 1))
        det_below = _make_detections(xyxy_below, ids)

        # Frame 2: all persons above the line
        xyxy_above = [[50 * i, 160, 50 * i + 40, 200] for i in range(n)]
        det_above = _make_detections(xyxy_above, ids)

        counter.update(det_below)
        events = counter.update(det_above)

        assert len(events) == n, f"Expected {n} crossing events, got {len(events)}"
        assert counter.in_count == n
        assert counter.current_count == n

    def test_many_detections_no_crossing(self, counter):
        """Many detections all on the same side → no events."""
        n = 15
        xyxy = [[50 * i, 280, 50 * i + 40, 320] for i in range(n)]
        ids = list(range(1, n + 1))

        counter.update(_make_detections(xyxy, ids))
        # Slightly different positions, same side
        xyxy2 = [[50 * i + 5, 270, 50 * i + 45, 310] for i in range(n)]
        events = counter.update(_make_detections(xyxy2, ids))

        assert len(events) == 0


class TestMixedDirections:
    def test_mixed_in_out(self, counter):
        """Some persons cross IN, others cross OUT simultaneously."""
        # Person 1: below → above (IN)
        # Person 2: above → below (OUT)

        det_frame1 = _make_detections(
            [[100, 280, 140, 320], [300, 160, 340, 200]],
            [1, 2],
        )
        det_frame2 = _make_detections(
            [[100, 160, 140, 200], [300, 280, 340, 320]],
            [1, 2],
        )

        counter.update(det_frame1)
        events = counter.update(det_frame2)

        directions = {e.person_id: e.direction for e in events}
        assert len(events) == 2
        assert directions[1] == "IN"
        assert directions[2] == "OUT"
        assert counter.current_count == 0  # 1 IN + 1 OUT = 0
