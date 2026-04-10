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

    # 2) DB 초기화 — PostgreSQL
    import server.models  # noqa: F401 — 모델을 Base.metadata에 등록
    from server.database import Base, engine

    Base.metadata.create_all(bind=engine)
    logger.info("PostgreSQL tables initialized")

    # 3) Repository 팩토리 → app.state.repo
    from server.repositories import get_repository

    app.state.repo = get_repository()
    logger.info("Repository initialized (backend=postgres)")

    # 4) AnimalService
    from server.services.animal.animal_service import animal_service

    animal_service.warmup()
    app.state.animal_service = animal_service

    # 7) PlantService
    from server.services.plant.plant_service import plant_service

    plant_service.warmup()
    app.state.plant_service = plant_service

    # 8) LettuceService
    from server.services.plant.lettuce_service import lettuce_service

    lettuce_service.warmup()
    app.state.lettuce_service = lettuce_service

    logger.info("Server ready.")
    yield
    logger.info("Shutting down...")



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
from server.routers import animal, lettuce, plant  # noqa: E402

app.include_router(animal.router, prefix="/api")
app.include_router(plant.router, prefix="/api")
app.include_router(lettuce.router, prefix="/api")
