import logging
import os
import shutil

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from server.config import settings
from server.schemas import (
    PersonCreate,
    PersonDetailResponse,
    PersonResponse,
    PersonUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["persons"])


@router.post("/persons", response_model=PersonResponse, status_code=201)
def create_person(body: PersonCreate, request: Request):
    repo = request.app.state.repo
    seq_num = repo.seq.next_person_number()
    name = f"person{seq_num}"
    person = repo.person.create(
        name=name,
        display_name=body.display_name,
        phone=body.phone,
        address=body.address,
        extra_info=body.extra_info,
    )
    logger.info(f"Person created: {name} (id={person.id})")
    return person


@router.get("/persons", response_model=list[PersonResponse])
def list_persons(request: Request):
    repo = request.app.state.repo
    return repo.person.list_all()


@router.get("/persons/{person_id}", response_model=PersonDetailResponse)
def get_person(person_id: int, request: Request):
    repo = request.app.state.repo
    person = repo.person.get(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.put("/persons/{person_id}", response_model=PersonResponse)
def update_person(person_id: int, body: PersonUpdate, request: Request):
    repo = request.app.state.repo
    update_data = body.model_dump(exclude_unset=True)
    person = repo.person.update(person_id, **update_data)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    logger.info(f"Person updated: id={person_id}")
    return person


@router.delete("/persons/{person_id}", status_code=204)
def delete_person(person_id: int, request: Request):
    repo = request.app.state.repo
    person = repo.person.get(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # face_db 디렉토리 내 해당 인물 이미지 파일 삭제
    person_face_dir = os.path.join(settings.FACE_DB_PATH, person.name)
    if os.path.isdir(person_face_dir):
        shutil.rmtree(person_face_dir)
        logger.info(f"Deleted face_db directory: {person_face_dir}")

    repo.person.delete(person_id)
    logger.info(f"Person deleted: id={person_id}, name={person.name}")
    return Response(status_code=204)
