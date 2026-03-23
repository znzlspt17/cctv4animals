"""공통 Fixture — mock 기반 테스트 환경."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from server.main import app

# ── Mock Person 객체 생성 헬퍼 ──


def _make_person(
    id: int = 1,
    name: str = "person1",
    display_name: str | None = None,
    phone: str | None = None,
    address: str | None = None,
    extra_info: dict | None = None,
):
    """from_attributes=True 호환 mock person 객체."""
    p = MagicMock()
    p.id = id
    p.name = name
    p.display_name = display_name
    p.phone = phone
    p.address = address
    p.extra_info = extra_info
    p.created_at = datetime(2025, 1, 1, 0, 0, 0)
    p.updated_at = datetime(2025, 1, 1, 0, 0, 0)
    p.face_images = []
    return p


def _make_face_image(
    id: int = 1,
    person_id: int = 1,
    image_path: str = "face_db/1/test.jpg",
    capture_condition: str | None = None,
):
    fi = MagicMock()
    fi.id = id
    fi.person_id = person_id
    fi.image_path = image_path
    fi.capture_condition = capture_condition
    fi.created_at = datetime(2025, 1, 1, 0, 0, 0)
    return fi


def _make_log_entry(
    id: int = 1,
    person_id: int | None = 1,
    person_name: str | None = "person1",
    confidence: float = 0.95,
    snapshot_path: str | None = None,
):
    log = MagicMock()
    log.id = id
    log.person_id = person_id
    log.confidence = confidence
    log.snapshot_path = snapshot_path
    log.recognized_at = datetime(2025, 1, 1, 12, 0, 0)
    # person relation
    person_mock = MagicMock()
    person_mock.name = person_name
    log.person = person_mock
    log.person_name = person_name
    return log


# ── Fixtures ──


@pytest.fixture
def mock_repo():
    """property-based AbstractRepository mock.

    repo.person, repo.face_image, repo.recognition_log, repo.alert_rule, repo.seq
    모두 MagicMock 으로 설정.
    """
    repo = MagicMock()
    # sub-repo 기본 반환값 설정
    repo.person.create.return_value = _make_person()
    repo.person.get.return_value = _make_person()
    repo.person.list_all.return_value = [
        _make_person(1, "person1"),
        _make_person(2, "person2"),
    ]
    repo.person.update.return_value = _make_person()
    repo.person.delete.return_value = None

    repo.face_image.create.return_value = _make_face_image()
    repo.face_image.get_by_person.return_value = [_make_face_image()]
    repo.face_image.get_all_embeddings.return_value = []
    repo.face_image.delete_by_person.return_value = None

    repo.recognition_log.create.return_value = _make_log_entry()
    repo.recognition_log.query.return_value = []
    repo.recognition_log.get_stats.return_value = {"total_logs": 0, "person_stats": []}
    repo.recognition_log.cleanup.return_value = 0
    repo.recognition_log.is_duplicate_log.return_value = False

    repo.alert_rule.get_active_rules.return_value = []
    repo.alert_rule.upsert.return_value = MagicMock()
    repo.alert_rule.get_by_person.return_value = []
    repo.alert_rule.delete.return_value = None

    repo.seq.next_person_number.return_value = 1
    return repo


@pytest.fixture
def mock_face_service():
    """FaceService mock."""
    svc = MagicMock()
    svc.search_face.return_value = []
    svc.register_face.return_value = {
        "face_image_id": 1,
        "person_id": 1,
        "message": "등록 완료",
    }
    svc.register_multi_angle.return_value = {
        "person_id": 1,
        "registered": [
            {"face_image_id": 1, "person_id": 1, "message": "등록 완료"},
        ],
        "failed": [],
    }
    return svc


@pytest.fixture
def client(mock_repo, mock_face_service):
    """TestClient with mocked dependencies injected into app.state."""
    app.state.repo = mock_repo
    app.state.face_service = mock_face_service
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def dummy_image_bytes():
    """100x100 검정 이미지 JPEG bytes."""
    import cv2
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
