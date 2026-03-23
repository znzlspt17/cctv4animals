"""인물 CRUD API 테스트 — POST/GET/PUT/DELETE /api/persons."""

from tests.conftest import _make_person

# ── POST /api/persons ──


def test_create_person(client, mock_repo):
    resp = client.post("/api/persons", json={})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "person1"
    mock_repo.seq.next_person_number.assert_called_once()
    mock_repo.person.create.assert_called_once()


def test_create_person_with_display_name(client, mock_repo):
    mock_repo.person.create.return_value = _make_person(
        id=2,
        name="person2",
        display_name="홍길동",
    )
    mock_repo.seq.next_person_number.return_value = 2
    resp = client.post("/api/persons", json={"display_name": "홍길동"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["display_name"] == "홍길동"


# ── GET /api/persons ──


def test_list_persons(client, mock_repo):
    resp = client.get("/api/persons")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    mock_repo.person.list_all.assert_called_once()


# ── GET /api/persons/{person_id} ──


def test_get_person(client, mock_repo):
    resp = client.get("/api/persons/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 1
    mock_repo.person.get.assert_called_once_with(1)


def test_get_person_not_found(client, mock_repo):
    mock_repo.person.get.return_value = None
    resp = client.get("/api/persons/999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Person not found"


# ── PUT /api/persons/{person_id} ──


def test_update_person(client, mock_repo):
    updated = _make_person(id=1, name="person1", display_name="수정됨")
    mock_repo.person.update.return_value = updated
    resp = client.put("/api/persons/1", json={"display_name": "수정됨"})
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "수정됨"
    mock_repo.person.update.assert_called_once()


def test_update_person_not_found(client, mock_repo):
    mock_repo.person.update.return_value = None
    resp = client.put("/api/persons/999", json={"display_name": "없는사람"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Person not found"


# ── DELETE /api/persons/{person_id} ──


def test_delete_person(client, mock_repo):
    resp = client.delete("/api/persons/1")
    assert resp.status_code == 204
    mock_repo.person.delete.assert_called_once_with(1)


def test_delete_person_not_found(client, mock_repo):
    mock_repo.person.get.return_value = None
    resp = client.delete("/api/persons/999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Person not found"
