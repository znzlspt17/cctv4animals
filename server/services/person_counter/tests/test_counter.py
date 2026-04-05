"""Unit tests for LineCrossCounter (counter.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers — build minimal sv.Detections-compatible objects
# ---------------------------------------------------------------------------


def _make_detections(xyxy_list, tracker_ids, confidences=None):
    """Build a mock detections object accepted by LineCrossCounter.update()."""
    det = MagicMock()
    det.xyxy = np.array(xyxy_list, dtype=np.float64)
    det.tracker_id = (
        np.array(tracker_ids, dtype=np.int64) if tracker_ids is not None else None
    )
    if confidences is not None:
        det.confidence = np.array(confidences, dtype=np.float64)
    else:
        det.confidence = np.ones(len(xyxy_list), dtype=np.float64) * 0.9
    det.__len__ = lambda self: len(xyxy_list)
    return det


@pytest.fixture
def counter():
    """Create a LineCrossCounter with a horizontal line at y=240."""
    with patch("counter.settings") as mock_s:
        mock_s.LINE_START_X = 0
        mock_s.LINE_START_Y = 240
        mock_s.LINE_END_X = 640
        mock_s.LINE_END_Y = 240
        mock_s.LINE_DIRECTION = "horizontal"
        from counter import LineCrossCounter

        return LineCrossCounter()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLineCrossIn:
    """test_line_cross_in: movement from below line → above line → IN event."""

    def test_line_cross_in(self, counter):
        # Arrange: person starts below line (cy=300) then moves above (cy=180)
        det_below = _make_detections(
            xyxy_list=[[100, 280, 200, 320]],  # cy = 300
            tracker_ids=[1],
        )
        det_above = _make_detections(
            xyxy_list=[[100, 160, 200, 200]],  # cy = 180
            tracker_ids=[1],
        )

        # Act: first frame sets position, second frame crosses
        events1 = counter.update(det_below)
        events2 = counter.update(det_above)

        # Assert
        assert len(events1) == 0, "First frame should not produce events"
        assert len(events2) == 1, "Second frame should produce 1 IN event"
        assert events2[0].direction == "IN"
        assert events2[0].person_id == 1
        assert events2[0].count_change == 1


class TestLineCrossOut:
    """test_line_cross_out: movement from above line → below line → OUT event."""

    def test_line_cross_out(self, counter):
        # Arrange: person starts above line (cy=180) then moves below (cy=300)
        det_above = _make_detections(
            xyxy_list=[[100, 160, 200, 200]],  # cy = 180
            tracker_ids=[1],
        )
        det_below = _make_detections(
            xyxy_list=[[100, 280, 200, 320]],  # cy = 300
            tracker_ids=[1],
        )

        # Act
        counter.update(det_above)
        events = counter.update(det_below)

        # Assert
        assert len(events) == 1
        assert events[0].direction == "OUT"
        assert events[0].count_change == -1


class TestDuplicatePrevention:
    """test_duplicate_prevention: same tracker_id crossing twice → only 1 event."""

    def test_duplicate_prevention(self, counter):
        # First crossing
        counter.update(_make_detections([[100, 280, 200, 320]], [1]))
        events1 = counter.update(_make_detections([[100, 160, 200, 200]], [1]))
        assert len(events1) == 1

        # Second crossing attempt (same ID goes back below and above again)
        counter.update(_make_detections([[100, 280, 200, 320]], [1]))
        events2 = counter.update(_make_detections([[100, 160, 200, 200]], [1]))
        assert len(events2) == 0, "Duplicate crossing should be prevented"


class TestNoCrossing:
    """test_no_crossing: person stays on one side → no events."""

    def test_no_crossing(self, counter):
        # Both positions are below the line
        det1 = _make_detections([[100, 280, 200, 320]], [1])  # cy=300
        det2 = _make_detections([[100, 260, 200, 300]], [1])  # cy=280

        counter.update(det1)
        events = counter.update(det2)

        assert len(events) == 0, "No line crossing → no events"


class TestCurrentCount:
    """test_current_count: IN/OUT events update current_count correctly."""

    def test_current_count(self, counter):
        # Person 1 crosses IN (below → above)
        counter.update(_make_detections([[100, 280, 200, 320]], [1]))
        counter.update(_make_detections([[100, 160, 200, 200]], [1]))
        assert counter.current_count == 1
        assert counter.in_count == 1

        # Person 2 crosses OUT (above → below)
        counter.update(_make_detections([[300, 160, 400, 200]], [2]))
        counter.update(_make_detections([[300, 280, 400, 320]], [2]))
        assert counter.current_count == 0
        assert counter.out_count == 1

        # Person 3 crosses IN
        counter.update(_make_detections([[500, 280, 600, 320]], [3]))
        counter.update(_make_detections([[500, 160, 600, 200]], [3]))
        assert counter.current_count == 1
        assert counter.in_count == 2
