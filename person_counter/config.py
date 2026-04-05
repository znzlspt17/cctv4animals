"""Settings class with .env loading and singleton instance."""

from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env",
        env_file_encoding="utf-8",
    )

    # 영상 소스
    VIDEO_SOURCE: str = "0"
    CAMERA_ID: str = "cam_01"

    # 카운팅 라인
    LINE_START_X: int = 0
    LINE_START_Y: int = 360
    LINE_END_X: int = 1280
    LINE_END_Y: int = 360
    LINE_DIRECTION: str = "horizontal"

    # 프레임 건너뛰기
    FRAME_SKIP: int = 10

    # ROI
    ROI_X: int = 0
    ROI_Y: int = 100
    ROI_W: int = 1280
    ROI_H: int = 520

    # DB
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_NAME: str = "people_counter"
    DB_USER: str = "root"
    DB_PASSWORD: str = ""

    # 알림
    KAKAO_ACCESS_TOKEN: str = ""
    KAKAO_REFRESH_TOKEN: str = ""
    ALERT_THRESHOLD: int = 50
    ALERT_COOLDOWN_SEC: int = 300

    # 대시보드
    DASHBOARD_PORT: int = 8000

    # YOLO
    YOLO_MODEL: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5


settings = Settings()

# ---- Loguru file rotation ----
_log_dir = Path(__file__).resolve().parent / "logs"
_log_dir.mkdir(exist_ok=True)
logger.add(
    str(_log_dir / "app.log"),
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)

logger.info(
    "Settings loaded: CAMERA_ID={}, VIDEO_SOURCE={}",
    settings.CAMERA_ID,
    settings.VIDEO_SOURCE,
)
