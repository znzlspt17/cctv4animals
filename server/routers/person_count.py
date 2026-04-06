import logging

from fastapi import APIRouter, Request

router = APIRouter(tags=["person-count"])
logger = logging.getLogger(__name__)


@router.get("/count/status")
async def count_status(request: Request):
    """카운팅 서비스 상태 및 현재 재실 인원 반환."""
    svc = request.app.state.person_counter_service
    return svc.get_stats()


@router.get("/count/events")
async def count_events(request: Request, limit: int = 100):
    """최근 라인 크로싱 이벤트 목록 반환."""
    repo = request.app.state.repo
    from server.config import settings
    events = repo.tracking_event.get_events(settings.PERSON_CAMERA_ID, limit=limit)
    return [
        {
            "id": e.id,
            "camera_id": e.camera_id,
            "tracker_id": e.tracker_id,
            "direction": e.direction,
            "count_change": e.count_change,
            "current_count": e.current_count,
            "confidence": e.confidence,
            "snapshot_path": e.snapshot_path,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in events
    ]


@router.post("/count/reset")
async def count_reset(request: Request):
    """카운팅 추적 상태 초기화 (재시작과 다름 — 현재 카운트를 0으로)."""
    svc = request.app.state.person_counter_service
    svc._current_count = 0
    svc._in_count = 0
    svc._out_count = 0
    logger.info("PersonCounter: count manually reset to 0")
    return {"message": "카운트 초기화 완료"}


@router.post("/count/camera/pause")
async def camera_pause(request: Request):
    """웹캠을 일시적으로 해제한다 (다른 앱이 웹캠을 사용해야 할 때 호출)."""
    svc = request.app.state.person_counter_service
    ok = svc.pause_camera(timeout=5.0)
    if ok:
        return {"message": "카메라 해제됨 (paused)", "paused": True}
    return {"message": "pause 타임아웃 — 카메라가 여전히 사용 중일 수 있음", "paused": False}


@router.post("/count/camera/resume")
async def camera_resume(request: Request):
    """pause 상태에서 웹캠을 재취득한다."""
    svc = request.app.state.person_counter_service
    svc.resume_camera()
    return {"message": "카메라 재개됨 (resumed)"}
