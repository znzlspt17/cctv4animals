"""Unit tests for database CRUD functions (database.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import get_current_count, get_events, get_last_state, save_event


class TestSaveAndGetEvent:
    def test_save_and_get_event(self, db_session):
        """save_event → get_events should return the saved event."""
        # Arrange
        event_data = {
            "person_id": 1,
            "camera_id": "cam_test",
            "direction": "IN",
            "count_change": 1,
            "current_count": 1,
            "confidence": 0.9,
            "bbox_x": 100,
            "bbox_y": 100,
            "bbox_w": 100,
            "bbox_h": 100,
        }

        # Act
        saved = save_event(db_session, event_data)
        events = get_events(db_session, "cam_test")

        # Assert
        assert saved.id is not None
        assert saved.direction == "IN"
        assert len(events) == 1
        assert events[0].person_id == 1

    def test_get_current_count(self, db_session):
        """get_current_count returns the latest current_count."""
        save_event(
            db_session,
            {
                "person_id": 1,
                "camera_id": "cam_test",
                "direction": "IN",
                "count_change": 1,
                "current_count": 1,
            },
        )
        save_event(
            db_session,
            {
                "person_id": 2,
                "camera_id": "cam_test",
                "direction": "IN",
                "count_change": 1,
                "current_count": 2,
            },
        )

        count = get_current_count(db_session, "cam_test")
        assert count == 2

    def test_get_current_count_no_events(self, db_session):
        """get_current_count returns 0 when no events exist."""
        count = get_current_count(db_session, "cam_test")
        assert count == 0

    def test_get_events_with_limit(self, db_session):
        """get_events respects the limit parameter."""
        for i in range(5):
            save_event(
                db_session,
                {
                    "person_id": i,
                    "camera_id": "cam_test",
                    "direction": "IN",
                    "count_change": 1,
                    "current_count": i + 1,
                },
            )

        events = get_events(db_session, "cam_test", limit=3)
        assert len(events) == 3

    def test_get_last_state(self, db_session):
        """get_last_state returns aggregated state."""
        save_event(
            db_session,
            {
                "person_id": 1,
                "camera_id": "cam_test",
                "direction": "IN",
                "count_change": 1,
                "current_count": 1,
            },
        )
        save_event(
            db_session,
            {
                "person_id": 2,
                "camera_id": "cam_test",
                "direction": "OUT",
                "count_change": -1,
                "current_count": 0,
            },
        )

        state = get_last_state(db_session, "cam_test")
        assert state["current_count"] == 0
        assert state["in_count"] == 1
        assert state["out_count"] == 1
        assert 1 in state["recent_person_ids"]
        assert 2 in state["recent_person_ids"]

    def test_get_last_state_empty(self, db_session):
        """get_last_state returns zeros when no events exist."""
        state = get_last_state(db_session, "cam_test")
        assert state["current_count"] == 0
        assert state["in_count"] == 0
        assert state["out_count"] == 0
