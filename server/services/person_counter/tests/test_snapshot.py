"""Unit tests for SnapshotManager (snapshot.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from counter import CountEvent


@pytest.fixture
def event():
    return CountEvent(
        person_id=42,
        direction="IN",
        count_change=1,
        confidence=0.95,
        bbox=(100.0, 100.0, 200.0, 200.0),
    )


@pytest.fixture
def frame():
    """640×480 frame with non-zero pixels so cv2.imwrite produces a real file."""
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    f[100:200, 100:200] = 128  # grey patch inside bbox area
    return f


class TestSaveCreatesFile:
    def test_save_creates_file(self, tmp_path, frame, event):
        """save() should create a full-frame JPEG file."""
        from snapshot import SnapshotManager

        mgr = SnapshotManager(base_dir=str(tmp_path))
        path = mgr.save(frame, event, camera_id="cam_test")

        assert os.path.isfile(path), f"Snapshot file not found: {path}"
        assert path.endswith(".jpg")


class TestSaveCreatesCrop:
    def test_save_creates_crop(self, tmp_path, frame, event):
        """save() should also create a _crop.jpg file."""
        from snapshot import SnapshotManager

        mgr = SnapshotManager(base_dir=str(tmp_path))
        full_path = mgr.save(frame, event, camera_id="cam_test")

        # Crop file lives in the same directory with _crop suffix
        dir_path = os.path.dirname(full_path)
        files = os.listdir(dir_path)
        crop_files = [f for f in files if "_crop.jpg" in f]
        assert len(crop_files) == 1, f"Expected 1 crop file, found {crop_files}"


class TestDirectoryCreation:
    def test_directory_creation(self, tmp_path, frame, event):
        """save() should auto-create camera_id/date subdirectories."""
        from snapshot import SnapshotManager

        mgr = SnapshotManager(base_dir=str(tmp_path))
        full_path = mgr.save(frame, event, camera_id="cam_test")

        # Verify the nested directory structure: base/cam_test/YYYYMMDD/
        parts = Path(full_path).relative_to(tmp_path).parts
        assert len(parts) >= 3, f"Expected nested dirs, got {parts}"
        assert parts[0] == "cam_test"
        assert parts[1].isdigit() and len(parts[1]) == 8  # YYYYMMDD
