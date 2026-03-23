import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv(
        "LOG_FORMAT", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    log_file_enabled = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/app.log")
    log_file_max_bytes = int(os.getenv("LOG_FILE_MAX_BYTES", "10485760"))
    log_file_backup_count = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    formatter = logging.Formatter(log_format)

    # Console handler (always)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (optional)
    if log_file_enabled:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=log_file_max_bytes,
            backupCount=log_file_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
