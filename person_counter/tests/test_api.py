"""Integration tests for FastAPI dashboard endpoints (dashboard/app.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client():
    """Create a TestClient with DB dependencies mocked."""
    # Mock the DB dependency and init_db to avoid real MySQL connection
    mock_session = MagicMock()

    # Mock get_last_state to return safe defaults
    with (
        patch("dashboard.app.init_db"),
        patch(
            "dashboard.app.get_last_state",
            return_value={
                "current_count": 5,
                "in_count": 10,
                "out_count": 5,
            },
        ),
        patch("dashboard.app.get_events", return_value=[]),
        patch("dashboard.app.get_db") as mock_get_db,
    ):
        mock_get_db.return_value = iter([mock_session])

        from dashboard.app import app

        with TestClient(app) as c:
            yield c


class TestIndexPage:
    def test_index_page(self, client):
        """GET / → 200 HTML response."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestApiStatus:
    def test_api_status(self, client):
        """GET /api/status → 200 with correct keys."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "camera_id" in data
        assert "current_count" in data
        assert "in_count" in data
        assert "out_count" in data

    def test_api_status_values(self, client):
        """GET /api/status returns mocked values."""
        resp = client.get("/api/status")
        data = resp.json()
        assert data["current_count"] == 5
        assert data["in_count"] == 10
        assert data["out_count"] == 5


class TestApiEvents:
    def test_api_events(self, client):
        """GET /api/events → 200 with a list."""
        resp = client.get("/api/events")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestApiStats:
    def test_api_stats(self, client):
        """GET /api/stats → 200 with correct schema."""
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "fps" in data
        assert "inference_ms" in data
        assert "tracking_ms" in data
        assert "gpu_memory_used_mb" in data
        assert "gpu_memory_total_mb" in data

    def test_api_stats_defaults(self, client):
        """Without performance monitor, stats return zeros."""
        resp = client.get("/api/stats")
        data = resp.json()
        assert data["fps"] == 0.0
