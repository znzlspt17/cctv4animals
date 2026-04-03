import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from server.config import Settings
from server.schemas import RecognizeResponse, RegisterResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recognition"])


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(request: Request, file: UploadFile = File(...)):
    """이미지 전송 → 얼굴 인식 결과 반환."""
    repo = request.app.state.repo
    face_svc = request.app.state.face_service

    image_bytes = await file.read()
    results = face_svc.search_face(image_bytes, repo)

    # 인식 로그 기록 (매칭된 얼굴만)
    settings = Settings()
    if settings.ENABLE_RECOGNITION_LOG:
        for r in results:
            if r["person_id"] is not None:
                try:
                    repo.recognition_log.create(
                        person_id=r["person_id"],
                        confidence=r["confidence"],
                        snapshot_path=None,
                    )
                except Exception as e:
                    logger.warning("Failed to save recognition log: %s", e)

    return {"results": results}


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    request: Request,
    file: UploadFile = File(...),
    person_id: int | None = Form(None),
    capture_condition: str | None = Form(None),
    display_name: str | None = Form(None),
    phone: str | None = Form(None),
    address: str | None = Form(None),
):
    """이미지 + (선택적 person_id) → 얼굴 등록. person_id 없으면 새 인물 자동 생성."""
    repo = request.app.state.repo
    face_svc = request.app.state.face_service

    image_bytes = await file.read()

    # person_id 없으면 새 인물 생성
    auto_created = False
    if person_id is None:
        seq_num = repo.seq.next_person_number()
        person_name = f"person{seq_num}"
        person = repo.person.create(
            name=person_name,
            display_name=display_name,
            phone=phone,
            address=address,
        )
        person_id = person.id
        person_name = person.name
        auto_created = True
        logger.info("Auto-created person: %s (id=%d)", person_name, person_id)
    else:
        person = repo.person.get(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        person_name = person.name

    try:
        result = face_svc.register_face(
            image_bytes=image_bytes,
            person_id=person_id,
            repo=repo,
            capture_condition=capture_condition,
        )
    except ValueError as e:
        # 등록 실패 시 자동 생성된 Person 롤백
        if auto_created:
            try:
                face_svc.delete_person_faces(person_id, repo)
                repo.person.delete(person_id)
                logger.info("Rolled back auto-created person_id=%d", person_id)
            except Exception as rollback_err:
                logger.warning(
                    "Rollback failed for person_id=%d: %s",
                    person_id,
                    rollback_err,
                )
        msg = str(e)
        if msg.startswith("DUPLICATE:"):
            raise HTTPException(status_code=409, detail=msg.split(":", 1)[1])
        raise HTTPException(status_code=400, detail=msg.split(":", 1)[-1])

    return RegisterResponse(
        person_id=person_id,
        person_name=person_name,
        face_image_id=result["face_image_id"],
        message=result["message"],
    )


@router.post("/register/multi-angle")
async def register_multi_angle(
    request: Request,
    files: list[UploadFile] = File(...),
    person_id: int | None = Form(None),
):
    """여러 각도/조명 이미지를 한 번에 등록."""
    repo = request.app.state.repo
    face_svc = request.app.state.face_service

    if not files:
        raise HTTPException(status_code=400, detail="이미지를 1장 이상 업로드하세요")

    # person_id 없으면 새 인물 생성
    auto_created = False
    if person_id is None:
        seq_num = repo.seq.next_person_number()
        person_name = f"person{seq_num}"
        person = repo.person.create(name=person_name)
        person_id = person.id
        auto_created = True
        logger.info(
            "Auto-created person for multi-angle: %s (id=%d)", person_name, person_id
        )

    images_bytes = [await f.read() for f in files]

    try:
        result = face_svc.register_multi_angle(
            images_list=images_bytes,
            repo=repo,
            person_id=person_id,
        )
    except Exception as e:
        # 전부 실패 시 자동 생성된 Person 롤백
        if auto_created:
            try:
                face_svc.delete_person_faces(person_id, repo)
                repo.person.delete(person_id)
                logger.info(
                    "Rolled back person_id=%d (multi)",
                    person_id,
                )
            except Exception as rollback_err:
                logger.warning(
                    "Rollback failed for person_id=%d: %s",
                    person_id,
                    rollback_err,
                )
        raise HTTPException(status_code=400, detail=str(e))

    # 부분 성공 시(등록된 얼굴이 하나도 없으면) Person 롤백
    registered = result.get("registered_count", len(result.get("results", [])))
    if auto_created and registered == 0:
        try:
            face_svc.delete_person_faces(person_id, repo)
            repo.person.delete(person_id)
            logger.info(
                "Rolled back person_id=%d (no faces)",
                person_id,
            )
        except Exception as rollback_err:
            logger.warning(
                "Rollback failed for person_id=%d: %s",
                person_id,
                rollback_err,
            )

    return result
