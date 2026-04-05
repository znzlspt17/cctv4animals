"""FPS and GPU performance monitoring module."""

from __future__ import annotations

import time
from collections import deque

from loguru import logger

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


class PerformanceMonitor:
    """Tracks FPS (moving average), inference/tracking time, and GPU memory."""

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = window_size
        self._frame_times: deque[float] = deque(maxlen=window_size)
        self._inference_times: deque[float] = deque(maxlen=window_size)
        self._tracking_times: deque[float] = deque(maxlen=window_size)
        self._frame_start: float = 0.0

        # GPU monitoring via pynvml
        self._gpu_handle = None
        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(self._gpu_handle)
                logger.info("PerformanceMonitor: GPU detected — {}", name)
            except pynvml.NVMLError as e:
                logger.warning("PerformanceMonitor: pynvml init failed — {}", e)
        else:
            logger.warning(
                "PerformanceMonitor: pynvml not available, GPU stats disabled"
            )

        logger.info("PerformanceMonitor initialized: window_size={}", window_size)

    def start_frame(self) -> None:
        """Record the start time of frame processing."""
        self._frame_start = time.perf_counter()

    def end_frame(self) -> None:
        """Record the end time and compute per-frame duration."""
        elapsed = time.perf_counter() - self._frame_start
        self._frame_times.append(elapsed)

    def record_inference(self, ms: float) -> None:
        """Record a YOLO inference duration in milliseconds."""
        self._inference_times.append(ms)

    def record_tracking(self, ms: float) -> None:
        """Record a ByteTrack update duration in milliseconds."""
        self._tracking_times.append(ms)

    def get_stats(self) -> dict[str, float]:
        """Return current performance statistics.

        Returns:
            Dictionary with fps, inference_ms, tracking_ms,
            gpu_memory_used_mb, gpu_memory_total_mb.
        """
        # FPS from moving average of frame times
        if self._frame_times:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        else:
            fps = 0.0

        inference_ms = (
            sum(self._inference_times) / len(self._inference_times)
            if self._inference_times
            else 0.0
        )
        tracking_ms = (
            sum(self._tracking_times) / len(self._tracking_times)
            if self._tracking_times
            else 0.0
        )

        gpu_mem_used = 0.0
        gpu_mem_total = 0.0
        if self._gpu_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_mem_used = mem_info.used / (1024 * 1024)
                gpu_mem_total = mem_info.total / (1024 * 1024)
            except pynvml.NVMLError:
                pass

        return {
            "fps": round(fps, 1),
            "inference_ms": round(inference_ms, 2),
            "tracking_ms": round(tracking_ms, 2),
            "gpu_memory_used_mb": round(gpu_mem_used, 1),
            "gpu_memory_total_mb": round(gpu_mem_total, 1),
        }
