"""인식/등록 API 테스트 — /api/recognize, /api/register, /api/register/multi-angle."""


# ── POST /api/recognize ──


def test_recognize_no_faces(client, mock_face_service, dummy_image_bytes):
    mock_face_service.search_face.return_value = []
    resp = client.post(
        "/api/recognize",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []


def test_recognize_with_match(
    client,
    mock_repo,
    mock_face_service,
    dummy_image_bytes,
):
    mock_face_service.search_face.return_value = [
        {
            "person_id": 1,
            "person_name": "person1",
            "display_name": None,
            "confidence": 0.92,
            "bbox": [10, 20, 50, 50],
            "alerts": [],
        },
    ]
    resp = client.post(
        "/api/recognize",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["person_id"] == 1
    assert results[0]["confidence"] == 0.92
    # 인식 로그 기록 확인
    mock_repo.recognition_log.create.assert_called_once()


# ── POST /api/register ──


def test_register_new_person(
    client,
    mock_repo,
    mock_face_service,
    dummy_image_bytes,
):
    """person_id 없이 등록 → 자동 생성."""
    resp = client.post(
        "/api/register",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["person_id"] == 1
    assert data["person_name"] == "person1"
    assert data["face_image_id"] == 1
    mock_repo.seq.next_person_number.assert_called_once()
    mock_repo.person.create.assert_called_once()
    mock_face_service.register_face.assert_called_once()


def test_register_existing_person(
    client,
    mock_repo,
    mock_face_service,
    dummy_image_bytes,
):
    """기존 person_id=1 에 얼굴 추가."""
    resp = client.post(
        "/api/register",
        data={"person_id": "1"},
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["person_id"] == 1
    mock_repo.person.get.assert_called_once_with(1)


def test_register_person_not_found(
    client,
    mock_repo,
    mock_face_service,
    dummy_image_bytes,
):
    mock_repo.person.get.return_value = None
    resp = client.post(
        "/api/register",
        data={"person_id": "999"},
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Person not found"


def test_register_duplicate(
    client,
    mock_repo,
    mock_face_service,
    dummy_image_bytes,
):
    mock_face_service.register_face.side_effect = ValueError(
        "DUPLICATE:이미 등록된 얼굴입니다 (매칭: person2, 유사도: 95.0%)"
    )
    resp = client.post(
        "/api/register",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 409
    assert "이미 등록된 얼굴" in resp.json()["detail"]


def test_register_validation_error(
    client,
    mock_repo,
    mock_face_service,
    dummy_image_bytes,
):
    mock_face_service.register_face.side_effect = ValueError(
        "NO_FACE:얼굴이 감지되지 않았습니다"
    )
    resp = client.post(
        "/api/register",
        files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "얼굴이 감지되지 않았습니다" in resp.json()["detail"]


# ── POST /api/register/multi-angle ──


def test_register_multi_angle(
    client,
    mock_repo,
    mock_face_service,
    dummy_image_bytes,
):
    mock_face_service.register_multi_angle.return_value = {
        "person_id": 1,
        "registered": [
            {"face_image_id": 1, "person_id": 1, "message": "등록 완료"},
            {"face_image_id": 2, "person_id": 1, "message": "등록 완료"},
            {"face_image_id": 3, "person_id": 1, "message": "등록 완료"},
        ],
        "failed": [],
    }
    files = [
        ("files", ("a.jpg", dummy_image_bytes, "image/jpeg")),
        ("files", ("b.jpg", dummy_image_bytes, "image/jpeg")),
        ("files", ("c.jpg", dummy_image_bytes, "image/jpeg")),
    ]
    resp = client.post("/api/register/multi-angle", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["person_id"] == 1
    assert len(data["registered"]) == 3
    mock_face_service.register_multi_angle.assert_called_once()
