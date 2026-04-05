"""Shared fixtures for People Counter tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# mock_settings — patches config.settings for all tests in a session
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    """Return a mock Settings object with safe defaults."""
    s = MagicMock()
    s.VIDEO_SOURCE = "0"
    s.CAMERA_ID = "cam_test"
    s.LINE_START_X = 0
    s.LINE_START_Y = 240
    s.LINE_END_X = 640
    s.LINE_END_Y = 240
    s.LINE_DIRECTION = "horizontal"
    s.FRAME_SKIP = 10
    s.ROI_X = 0
    s.ROI_Y = 0
    s.ROI_W = 640
    s.ROI_H = 480
    s.YOLO_MODEL = "yolov8n.pt"
    s.CONFIDENCE_THRESHOLD = 0.5
    s.DB_HOST = "localhost"
    s.DB_PORT = 5432
    s.DB_NAME = "test_db"
    s.DB_USER = "postgres"
    s.DB_PASSWORD = ""
    s.KAKAO_ACCESS_TOKEN = ""
    s.KAKAO_REFRESH_TOKEN = ""
    s.ALERT_THRESHOLD = 50
    s.ALERT_COOLDOWN_SEC = 300
    return s


# ---------------------------------------------------------------------------
# sample_frame — 640×480 black BGR frame
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_frame():
    """Return a 640×480 black BGR image (numpy array)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# sample_detections — minimal sv.Detections-like mock
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_detections():
    """Return a mock sv.Detections with 2 persons."""
    det = MagicMock()
    det.xyxy = np.array(
        [
            [100.0, 100.0, 200.0, 200.0],
            [300.0, 300.0, 400.0, 400.0],
        ]
    )
    det.confidence = np.array([0.9, 0.85])
    det.class_id = np.array([0, 0])
    det.tracker_id = np.array([1, 2])
    det.__len__ = lambda self: 2
    return det


# ---------------------------------------------------------------------------
# sample_count_event
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_count_event():
    """Return a CountEvent instance."""
    from counter import CountEvent

    return CountEvent(
        person_id=1,
        direction="IN",
        count_change=1,
        confidence=0.9,
        bbox=(100.0, 100.0, 200.0, 200.0),
    )


# ---------------------------------------------------------------------------
# db_session — SQLite in-memory session for database tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session():
    """Yield a SQLAlchemy session backed by in-memory SQLite."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()
