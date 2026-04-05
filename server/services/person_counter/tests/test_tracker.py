"""Unit tests for PersonTracker (tracker.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tracker():
    """Create a PersonTracker with FRAME_SKIP=10."""
    with patch("tracker.settings") as mock_s:
        mock_s.FRAME_SKIP = 10
        with patch("tracker.sv") as mock_sv:
            mock_sv.ByteTrack.return_value = MagicMock()
            from tracker import PersonTracker

            return PersonTracker()


class TestShouldDetect:
    def test_should_detect_first_frame(self, tracker):
        """frame_count=0 → should_detect returns True."""
        assert tracker.frame_count == 0
        assert tracker.should_detect() is True

    def test_should_detect_skip(self, tracker):
        """frame_count 1~9 → should_detect returns False."""
        for i in range(1, 10):
            tracker.frame_count = i
            assert tracker.should_detect() is False, (
                f"frame_count={i} should return False"
            )

    def test_should_detect_nth(self, tracker):
        """frame_count=10 → should_detect returns True."""
        tracker.frame_count = 10
        assert tracker.should_detect() is True

    def test_should_detect_multiples(self, tracker):
        """frame_count at multiples of FRAME_SKIP → True."""
        for n in [0, 10, 20, 30]:
            tracker.frame_count = n
            assert tracker.should_detect() is True


class TestIncrementFrame:
    def test_increment_frame(self, tracker):
        """increment_frame increases frame_count by 1."""
        assert tracker.frame_count == 0
        tracker.increment_frame()
        assert tracker.frame_count == 1
        tracker.increment_frame()
        assert tracker.frame_count == 2


class TestReset:
    def test_reset(self, tracker):
        """reset() sets frame_count back to 0 and resets byte_track."""
        tracker.frame_count = 42
        tracker.reset()
        assert tracker.frame_count == 0
        tracker.byte_track.reset.assert_called_once()


class TestUpdate:
    def test_update_calls_bytetrack(self, tracker):
        """update() delegates to byte_track.update_with_detections."""
        mock_det = MagicMock()
        tracker.update(mock_det)
        tracker.byte_track.update_with_detections.assert_called_once_with(mock_det)
