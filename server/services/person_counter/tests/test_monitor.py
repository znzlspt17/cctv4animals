"""Unit tests for PerformanceMonitor (monitor.py)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def monitor():
    """Create a PerformanceMonitor with pynvml disabled."""
    with patch("monitor._PYNVML_AVAILABLE", False):
        from monitor import PerformanceMonitor

        return PerformanceMonitor(window_size=10)


class TestFpsCalculation:
    def test_fps_calculation(self, monitor):
        """After several start/end cycles, fps should be > 0."""
        for _ in range(5):
            monitor.start_frame()
            time.sleep(0.01)  # ~10ms per frame → ~100 FPS
            monitor.end_frame()

        stats = monitor.get_stats()
        assert stats["fps"] > 0, f"FPS should be > 0, got {stats['fps']}"


class TestRecordInference:
    def test_record_inference(self, monitor):
        """record_inference stores values; get_stats reports the average."""
        monitor.record_inference(15.0)
        monitor.record_inference(25.0)

        stats = monitor.get_stats()
        assert stats["inference_ms"] == pytest.approx(20.0, abs=0.1)


class TestRecordTracking:
    def test_record_tracking(self, monitor):
        """record_tracking stores values correctly."""
        monitor.record_tracking(5.0)
        monitor.record_tracking(10.0)

        stats = monitor.get_stats()
        assert stats["tracking_ms"] == pytest.approx(7.5, abs=0.1)


class TestGetStatsEmpty:
    def test_get_stats_empty(self, monitor):
        """Initial state should return all zeros."""
        stats = monitor.get_stats()
        assert stats["fps"] == 0.0
        assert stats["inference_ms"] == 0.0
        assert stats["tracking_ms"] == 0.0
        assert stats["gpu_memory_used_mb"] == 0.0
        assert stats["gpu_memory_total_mb"] == 0.0
