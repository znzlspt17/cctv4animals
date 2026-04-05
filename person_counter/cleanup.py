"""Automatic cleanup of old snapshot files (>30 days)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
MAX_AGE_DAYS = 30


def cleanup_snapshots(
    directory: Path = SNAPSHOT_DIR, max_age_days: int = MAX_AGE_DAYS
) -> int:
    """Delete snapshot files older than max_age_days. Returns count of deleted files."""
    if not directory.exists():
        logger.info("Snapshot directory does not exist: {}", directory)
        return 0

    cutoff = datetime.now() - timedelta(days=max_age_days)
    deleted = 0

    for filepath in directory.iterdir():
        if not filepath.is_file():
            continue
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        if mtime < cutoff:
            filepath.unlink()
            logger.info(
                "Deleted old snapshot: {} (modified {})", filepath.name, mtime.date()
            )
            deleted += 1

    logger.info(
        "Cleanup complete: {} files deleted (threshold: {} days)", deleted, max_age_days
    )
    return deleted


if __name__ == "__main__":
    cleanup_snapshots()
