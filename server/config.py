from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL (원격)
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@100.95.34.69:5555/cctv?sslmode=disable"

    # FastAPI / Streamlit
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    STREAMLIT_PORT: int = 8501

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
