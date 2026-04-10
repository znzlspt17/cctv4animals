from __future__ import annotations

import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # PostgreSQL (원격)
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@100.95.34.69:5555/cctv?sslmode=disable"

    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    # 이미지 다운로드 URL 생성에 사용할 공개 IP/호스트 (비어 있으면 request.base_url 사용)
    FASTAPI_PUBLIC_HOST: str = "172.16.30.124:8000"

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

    # Result Publisher — 추론 이벤트 외부 전송 (빈 문자열이면 전송 비활성화)
    RESULT_PUBLISHER_BASE_URL: str = ""
    RESULT_PUBLISHER_TIMEOUT: float = 2.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
