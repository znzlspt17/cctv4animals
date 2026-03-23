import logging
from datetime import datetime

from fastapi import APIRouter, Query, Request

from server.config import settings
from server.schemas import LogResponse, LogStatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["logs"])


@router.get("/logs", response_model=list[LogResponse])
def query_logs(
    request: Request,
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    person_id: int | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    repo = request.app.state.repo
    logs = repo.recognition_log.query(
        start_date=start_date,
        end_date=end_date,
        person_id=person_id,
        page=page,
        page_size=page_size,
    )
    results = []
    for log_entry in logs:
        person_name = None
        if hasattr(log_entry, "person") and log_entry.person:
            person_name = log_entry.person.name
        elif hasattr(log_entry, "person_name"):
            person_name = log_entry.person_name
        results.append(
            LogResponse(
                id=log_entry.id,
                person_id=log_entry.person_id,
                person_name=person_name,
                confidence=log_entry.confidence,
                snapshot_path=log_entry.snapshot_path,
                recognized_at=log_entry.recognized_at,
            )
        )
    return results


@router.get("/logs/stats", response_model=LogStatsResponse)
def get_log_stats(request: Request):
    repo = request.app.state.repo
    stats = repo.recognition_log.get_stats()
    return LogStatsResponse(**stats)


@router.delete("/logs/cleanup")
def cleanup_logs(
    request: Request,
    retention_days: int = Query(None),
):
    if retention_days is None:
        retention_days = settings.LOG_RETENTION_DAYS
    repo = request.app.state.repo
    deleted_count = repo.recognition_log.cleanup(retention_days)
    logger.info(
        f"Log cleanup: deleted {deleted_count} records (retention={retention_days} days)"
    )
    return {"deleted_count": deleted_count}
