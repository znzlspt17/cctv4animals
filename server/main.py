import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) 로깅 초기화
    from logger import setup_logging

    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting DeepFace Live server...")

    # 2) DB 초기화 — DB_BACKEND 분기
    from server.config import settings

    if settings.DB_BACKEND == "mysql":
        from server.database import Base, engine

        import server.models  # noqa: F401 — 모델을 Base.metadata에 등록

        Base.metadata.create_all(bind=engine)
        _ensure_person_seq()
        logger.info("MySQL tables initialized")
    else:
        from server.repositories.redis_repo import RedisRepository

        repo = RedisRepository()
        repo.init_indexes()
        logger.info("Redis indexes initialized")

    # 3) Repository 팩토리 → app.state.repo
    from server.repositories import get_repository

    app.state.repo = get_repository()
    logger.info(f"Repository initialized (backend={settings.DB_BACKEND})")

    # 4) FaceService
    from server.services.face_service import face_service

    face_service.warmup()
    face_service.load_embedding_cache(app.state.repo)
    app.state.face_service = face_service

    logger.info("Server ready.")
    yield
    logger.info("Shutting down...")


def _ensure_person_seq():
    """Ensure person_name_seq table has initial row with id=1."""
    from server.database import SessionLocal
    from server.models import PersonNameSeq

    db = SessionLocal()
    try:
        row = db.query(PersonNameSeq).filter(PersonNameSeq.id == 1).first()
        if not row:
            db.add(PersonNameSeq(id=1, next_val=1))
            db.commit()
    finally:
        db.close()


app = FastAPI(title="DeepFace Live", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 글로벌 에러 핸들러
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "error_code": "VALIDATION_ERROR"},
    )


@app.exception_handler(ConnectionError)
async def connection_error_handler(request: Request, exc: ConnectionError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Service unavailable", "error_code": "DB_UNAVAILABLE"},
    )


# 라우터 등록
from server.routers import log, person, recognition  # noqa: E402

app.include_router(person.router, prefix="/api")
app.include_router(log.router, prefix="/api")
app.include_router(recognition.router, prefix="/api")
