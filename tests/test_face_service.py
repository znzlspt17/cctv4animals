"""FaceService 단위 테스트 — DeepFace 를 mock 하여 순수 로직 검증."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from server.services.face_service import FaceService


@pytest.fixture
def svc():
    """깨끗한 FaceService 인스턴스."""
    return FaceService()


# ── Warmup ──


@patch("server.services.face_service.DeepFace")
def test_warmup(mock_deepface, svc):
    svc.warmup()
    mock_deepface.represent.assert_called_once()


# ── Cosine Distance ──


def test_cosine_distance_identical(svc):
    a = np.array([1.0, 0.0, 0.0])
    dist = svc._cosine_distance(a, a)
    assert abs(dist) < 1e-6


def test_cosine_distance_orthogonal(svc):
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    dist = svc._cosine_distance(a, b)
    assert abs(dist - 1.0) < 1e-6


def test_cosine_distance_opposite(svc):
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    dist = svc._cosine_distance(a, b)
    assert abs(dist - 2.0) < 1e-6


# ── Check Duplicate ──


def test_check_duplicate_no_cache(svc):
    repo = MagicMock()
    svc._embedding_cache = []
    emb = np.random.randn(128).astype(np.float32)
    is_dup, person_id, distance = svc.check_duplicate(emb, repo)
    assert is_dup is False
    assert person_id is None
    assert distance == float("inf")


def test_check_duplicate_found(svc):
    """캐시에 유사 임베딩 존재 → 중복 판정."""
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    svc._embedding_cache = [
        {
            "person_id": 10,
            "person_name": "person10",
            "face_image_id": 100,
            "embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        },
    ]
    repo = MagicMock()
    with patch("server.services.face_service.settings") as mock_settings:
        mock_settings.RECOGNITION_THRESHOLD = 0.40
        is_dup, person_id, distance = svc.check_duplicate(emb, repo)
    assert is_dup is True
    assert person_id == 10
    assert distance < 0.01


def test_check_duplicate_not_found(svc):
    """캐시에 유사 임베딩 없음 → 중복 아님."""
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    svc._embedding_cache = [
        {
            "person_id": 20,
            "person_name": "person20",
            "face_image_id": 200,
            "embedding": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # orthogonal
        },
    ]
    repo = MagicMock()
    with patch("server.services.face_service.settings") as mock_settings:
        mock_settings.RECOGNITION_THRESHOLD = 0.40
        is_dup, person_id, distance = svc.check_duplicate(emb, repo)
    assert is_dup is False


def test_check_duplicate_exclude_person(svc):
    """exclude_person_id 적용 → 해당 person 제외하고 비교."""
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    svc._embedding_cache = [
        {
            "person_id": 5,
            "person_name": "person5",
            "face_image_id": 50,
            "embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        },
    ]
    repo = MagicMock()
    with patch("server.services.face_service.settings") as mock_settings:
        mock_settings.RECOGNITION_THRESHOLD = 0.40
        is_dup, person_id, distance = svc.check_duplicate(
            emb,
            repo,
            exclude_person_id=5,
        )
    # 유일한 candidate 가 exclude 되므로 빈 캐시와 동일
    assert is_dup is False
    assert person_id is None
    assert distance == float("inf")


# ── Embedding Cache ──


def test_load_embedding_cache(svc):
    repo = MagicMock()
    repo.face_image.get_all_embeddings.return_value = [
        {
            "person_id": 1,
            "face_image_id": 10,
            "embedding": np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes(),
        },
    ]
    person_mock = MagicMock()
    person_mock.id = 1
    person_mock.name = "person1"
    repo.person.list_all.return_value = [person_mock]

    svc.load_embedding_cache(repo)

    assert svc._cache_loaded is True
    assert len(svc._embedding_cache) == 1
    assert svc._embedding_cache[0]["person_id"] == 1
    assert svc._embedding_cache[0]["person_name"] == "person1"


def test_add_to_cache(svc):
    svc._embedding_cache = []
    emb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    svc._add_to_cache(
        person_id=3,
        person_name="person3",
        face_image_id=30,
        embedding=emb,
    )
    assert len(svc._embedding_cache) == 1
    assert svc._embedding_cache[0]["person_id"] == 3


def test_invalidate_cache_for_person(svc):
    svc._embedding_cache = [
        {
            "person_id": 1,
            "person_name": "person1",
            "face_image_id": 10,
            "embedding": np.zeros(3, dtype=np.float32),
        },
        {
            "person_id": 2,
            "person_name": "person2",
            "face_image_id": 20,
            "embedding": np.zeros(3, dtype=np.float32),
        },
        {
            "person_id": 1,
            "person_name": "person1",
            "face_image_id": 11,
            "embedding": np.zeros(3, dtype=np.float32),
        },
    ]
    svc._invalidate_cache_for_person(1)
    assert len(svc._embedding_cache) == 1
    assert svc._embedding_cache[0]["person_id"] == 2
