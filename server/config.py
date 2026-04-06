from __future__ import annotations

import json
import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # PostgreSQL (원격)
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@100.95.34.69:5555/cctv?sslmode=disable"

    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000

    # Face Recognition
    FACE_DB_PATH: str = "face_db"
    DEEPFACE_MODEL: str = "ArcFace"
    DEEPFACE_DETECTOR: str = "retinaface"
    DEEPFACE_DETECTOR_REALTIME: str = "retinaface"
    DEEPFACE_DISTANCE_METRIC: str = "cosine"

    # Registration conditions (R2~R4)
    FACE_MIN_CONFIDENCE: float = 0.90
    FACE_MIN_SIZE: int = 112
    FACE_BLUR_THRESHOLD: float = 100.0

    # Detection conditions (D2~D3)
    FACE_MIN_CONFIDENCE_REALTIME: float = 0.80
    FACE_MIN_SIZE_REALTIME: int = 56

    # Recognition
    RECOGNITION_THRESHOLD: float = 0.40
    DUPLICATE_THRESHOLD: float = 0.25
    ALLOW_FORCE_REGISTER: bool = False
    RECOGNITION_FRAME_SKIP: int = 3

    # Recognition Log
    ENABLE_RECOGNITION_LOG: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_FILE_ENABLED: bool = True
    LOG_FILE_PATH: str = "logs/app.log"
    LOG_FILE_MAX_BYTES: int = 10485760
    LOG_FILE_BACKUP_COUNT: int = 5
    LOG_RETENTION_DAYS: int = 30
    LOG_DEDUP_SECONDS: int = 10

    # Animal line crossing
    ANIMAL_LINE_START_X: int = 0
    ANIMAL_LINE_START_Y: int = 860
    ANIMAL_LINE_END_X: int = 1920
    ANIMAL_LINE_END_Y: int = 860

    # Person counter
    PERSON_VIDEO_SOURCE: str = "0"
    PERSON_CAMERA_ID: str = "cam_01"
    PERSON_LINE_START_X: int = 0
    PERSON_LINE_START_Y: int = 360
    PERSON_LINE_END_X: int = 1280
    PERSON_LINE_END_Y: int = 360
    PERSON_FRAME_SKIP: int = 10
    PERSON_ROI_X: int = 0
    PERSON_ROI_Y: int = 100
    PERSON_ROI_W: int = 1280
    PERSON_ROI_H: int = 520
    PERSON_YOLO_MODEL: str = "models/4people-yolo26n.pt"
    PERSON_CONFIDENCE_THRESHOLD: float = 0.5
    PERSON_SNAPSHOT_DIR: str = "snapshots"

    # 멀티카메라 설정 (JSON 배열 문자열)
    # 비어 있으면 위의 PERSON_* 단일 카메라 설정을 사용한다.
    # 예) CAMERAS_JSON='[{"camera_id":"cam_01","video_source":"0","label":"정문",...}]'
    CAMERAS_JSON: str = ""

    # Result Publisher — 추론 이벤트 외부 전송
    RESULT_PUBLISHER_BASE_URL: str = "http://172.16.15.43:8000"
    RESULT_PUBLISHER_TIMEOUT: float = 2.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


def get_camera_configs() -> list:
    """환경 변수에서 카메라 설정 목록을 반환한다.

    ``CAMERAS_JSON`` 이 설정되어 있으면 해당 JSON 배열을 파싱하며,
    비어 있으면 기존 ``PERSON_*`` 단일 카메라 설정으로 폴백한다.

    Returns:
        ``CameraConfig`` 인스턴스 리스트.
    """
    from server.services.person.camera_config import CameraConfig  # 지연 임포트 (순환 방지)

    if settings.CAMERAS_JSON.strip():
        try:
            raw_list: list[dict] = json.loads(settings.CAMERAS_JSON)
        except json.JSONDecodeError as exc:
            logger.error("CAMERAS_JSON 파싱 실패: %s — 단일 카메라 폴백", exc)
            raw_list = []

        if raw_list:
            configs = []
            for item in raw_list:
                try:
                    configs.append(CameraConfig(**item))
                except TypeError as exc:
                    logger.error("카메라 설정 항목 오류 (%s): %s — 건너뜀", item, exc)
            if configs:
                return configs

    # 폴백: 기존 단일 카메라 설정
    return [
        CameraConfig(
            camera_id=settings.PERSON_CAMERA_ID,
            video_source=settings.PERSON_VIDEO_SOURCE,
            line_start_x=settings.PERSON_LINE_START_X,
            line_start_y=settings.PERSON_LINE_START_Y,
            line_end_x=settings.PERSON_LINE_END_X,
            line_end_y=settings.PERSON_LINE_END_Y,
            frame_skip=settings.PERSON_FRAME_SKIP,
            roi_x=settings.PERSON_ROI_X,
            roi_y=settings.PERSON_ROI_Y,
            roi_w=settings.PERSON_ROI_W,
            roi_h=settings.PERSON_ROI_H,
            confidence_threshold=settings.PERSON_CONFIDENCE_THRESHOLD,
            snapshot_dir=settings.PERSON_SNAPSHOT_DIR,
        )
    ]
