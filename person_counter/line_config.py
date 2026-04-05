"""Counting line configuration UI module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from config import settings


class LineConfigurator:
    """Interactive OpenCV UI for setting the counting line via mouse drag."""

    _WINDOW_NAME = "Line Configurator — drag to draw, ENTER to confirm, ESC to cancel"

    def __init__(self, video_source: str | int | None = None) -> None:
        source = video_source if video_source is not None else settings.VIDEO_SOURCE
        # If the source looks like a device index, cast to int
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError(f"Cannot read frame from video source: {source}")

        self._frame: np.ndarray = frame.copy()
        self._display: np.ndarray = frame.copy()

        self._start_point: tuple[int, int] | None = None
        self._end_point: tuple[int, int] | None = None
        self._dragging: bool = False
        self._confirmed: bool = False

        logger.info(
            "LineConfigurator: captured frame {}x{} from source={}",
            frame.shape[1],
            frame.shape[0],
            source,
        )

    def _mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: object
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._start_point = (x, y)
            self._end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._end_point = (x, y)
            self._display = self._frame.copy()
            cv2.line(self._display, self._start_point, self._end_point, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self._end_point = (x, y)
            self._display = self._frame.copy()
            if self._start_point and self._end_point:
                cv2.line(
                    self._display, self._start_point, self._end_point, (0, 255, 0), 2
                )

    def run(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Open the interactive line-setting window.

        Returns:
            Tuple of (start_point, end_point) if confirmed, else None.
        """
        cv2.namedWindow(self._WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._WINDOW_NAME, self._mouse_callback)

        logger.info(
            "LineConfigurator UI started. Drag to set line, ENTER to confirm, ESC to cancel."
        )

        while True:
            cv2.imshow(self._WINDOW_NAME, self._display)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # Enter
                if self._start_point and self._end_point:
                    self._confirmed = True
                    self._save_to_env(self._start_point, self._end_point)
                    logger.info(
                        "Line confirmed: ({},{}) → ({},{})",
                        *self._start_point,
                        *self._end_point,
                    )
                break
            elif key == 27:  # ESC
                logger.info("Line configuration cancelled.")
                break

        cv2.destroyAllWindows()

        if self._confirmed and self._start_point and self._end_point:
            return (self._start_point, self._end_point)
        return None

    @staticmethod
    def _save_to_env(start: tuple[int, int], end: tuple[int, int]) -> None:
        """Update LINE_START/END coordinates in the .env file."""
        env_path = Path(__file__).resolve().parent / ".env"
        updates = {
            "LINE_START_X": str(start[0]),
            "LINE_START_Y": str(start[1]),
            "LINE_END_X": str(end[0]),
            "LINE_END_Y": str(end[1]),
        }

        lines: list[str] = []
        found_keys: set[str] = set()

        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
            new_lines: list[str] = []
            for line in lines:
                key = line.split("=", 1)[0].strip() if "=" in line else ""
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}")
                    found_keys.add(key)
                else:
                    new_lines.append(line)
            lines = new_lines

        # Append any keys that were not already in the file
        for key, value in updates.items():
            if key not in found_keys:
                lines.append(f"{key}={value}")

        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Updated .env: {}", updates)


def configure_line(
    video_source: str | int | None = None,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Convenience function to run the line configurator.

    Args:
        video_source: Optional video source override.

    Returns:
        Tuple of (start_point, end_point) if confirmed, else None.
    """
    configurator = LineConfigurator(video_source)
    return configurator.run()
