"""Repository 인터페이스 테스트 — 모듈 import 및 계약 확인."""

from server.repositories.base import (
    AbstractRepository,
    AlertRuleRepo,
    FaceImageRepo,
    PersonRepo,
    RecognitionLogRepo,
    SeqRepo,
)

# ── AbstractRepository 인터페이스 계약 ──


def test_abstract_repository_has_sub_repos():
    """AbstractRepository 가 5개 sub-repo property 를 정의하는지 확인."""
    expected_props = ["person", "face_image", "recognition_log", "alert_rule", "seq"]
    for prop_name in expected_props:
        assert hasattr(AbstractRepository, prop_name), f"Missing property: {prop_name}"
        attr = getattr(AbstractRepository, prop_name)
        assert isinstance(attr, property), f"{prop_name} should be a property"


def test_person_repo_interface():
    """PersonRepo ABC 메서드 확인."""
    methods = ["create", "get", "list_all", "update", "delete"]
    for m in methods:
        assert hasattr(PersonRepo, m), f"PersonRepo missing method: {m}"
        assert callable(getattr(PersonRepo, m))


def test_face_image_repo_interface():
    """FaceImageRepo ABC 메서드 확인."""
    methods = ["create", "get_by_person", "get_all_embeddings", "delete_by_person"]
    for m in methods:
        assert hasattr(FaceImageRepo, m), f"FaceImageRepo missing method: {m}"


def test_recognition_log_repo_interface():
    methods = ["create", "query", "get_stats", "cleanup", "is_duplicate_log"]
    for m in methods:
        assert hasattr(RecognitionLogRepo, m), f"RecognitionLogRepo missing method: {m}"


def test_alert_rule_repo_interface():
    methods = ["get_active_rules", "upsert", "get_by_person", "delete"]
    for m in methods:
        assert hasattr(AlertRuleRepo, m), f"AlertRuleRepo missing method: {m}"


def test_seq_repo_interface():
    assert hasattr(SeqRepo, "next_person_number")


# ── Postgres 모듈 import 확인 ──


def test_postgres_repo_importable():
    """postgres_repo 모듈이 import 가능한지 확인."""
    from server.repositories import postgres_repo  # noqa: F401

    assert hasattr(postgres_repo, "PostgresRepository")


# ── get_repository 팩토리 ──


def test_get_repository_factory_exists():
    """get_repository() 함수 존재 확인."""
    from server.repositories import get_repository

    assert callable(get_repository)


def test_abstract_repository_is_abstract():
    """AbstractRepository 를 직접 인스턴스화하면 TypeError."""
    import pytest

    with pytest.raises(TypeError):
        AbstractRepository()
