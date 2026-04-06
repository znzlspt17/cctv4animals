import logging

from fastapi import APIRouter, HTTPException, Request

from server.schemas import (
    CameraAggregateResponse,
    CameraConfigRequest,
    CameraStatusResponse,
)

router = APIRouter(tags=["person-count"])
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 전체 카메라 요약
# ──────────────────────────────────────────────────────────────

@router.get("/count/cameras")
async def list_cameras(request: Request):
    """등록된 모든 카메라의 상태 목록을 반환한다."""
    mgr = request.app.state.camera_manager
    return mgr.list_stats()


@router.post("/count/cameras", status_code=201, response_model=CameraStatusResponse)
async def add_camera(body: CameraConfigRequest, request: Request):
    """런타임에 새 카메라를 동적으로 추가한다."""
    from server.services.person.camera_config import CameraConfig

    mgr = request.app.state.camera_manager
    repo = request.app.state.repo
    cfg = CameraConfig(**body.model_dump())
    svc = mgr.add_camera(cfg, repo)
    return svc.get_stats()


@router.delete("/count/cameras/{camera_id}", status_code=200)
async def remove_camera(camera_id: str, request: Request):
    """실행 중인 카메라를 중지하고 목록에서 제거한다."""
    mgr = request.app.state.camera_manager
    removed = mgr.remove_camera(camera_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
    return {"camera_id": camera_id, "message": "카메라 제거됨"}


@router.get("/count/aggregate")
async def count_aggregate(request: Request):
    """전체 카메라의 합산 인원수를 반환한다."""
    mgr = request.app.state.camera_manager
    return mgr.aggregate_count()


# ──────────────────────────────────────────────────────────────
# 특정 카메라 조작
# ──────────────────────────────────────────────────────────────

@router.get("/count/cameras/{camera_id}/status")
async def camera_status(camera_id: str, request: Request):
    """특정 카메라의 상태 및 현재 재실 인원을 반환한다."""
    mgr = request.app.state.camera_manager
    svc = mgr.get(camera_id)
    if svc is None:
        raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
    return svc.get_stats()


@router.get("/count/cameras/{camera_id}/events")
async def camera_events(camera_id: str, request: Request, limit: int = 100):
    """특정 카메라의 최근 라인 크로싱 이벤트 목록을 반환한다."""
    mgr = request.app.state.camera_manager
    if mgr.get(camera_id) is None:
        raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
    repo = request.app.state.repo
    events = repo.tracking_event.get_events(camera_id, limit=limit)
    return [
        {
            "id": e["id"],
            "camera_id": e["camera_id"],
            "tracker_id": e["tracker_id"],
            "direction": e["direction"],
            "count_change": e["count_change"],
            "current_count": e["current_count"],
            "confidence": e["confidence"],
            "snapshot_path": e["snapshot_path"],
            "created_at": e["created_at"].isoformat() if e["created_at"] else None,
        }
        for e in events
    ]


@router.post("/count/cameras/{camera_id}/reset")
async def camera_reset(camera_id: str, request: Request):
    """특정 카메라의 카운트를 0으로 초기화한다."""
    mgr = request.app.state.camera_manager
    svc = mgr.get(camera_id)
    if svc is None:
        raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
    svc._current_count = 0
    svc._in_count = 0
    svc._out_count = 0
    logger.info("PersonCounter[%s]: count manually reset to 0", camera_id)
    return {"camera_id": camera_id, "message": "카운트 초기화 완료"}


@router.post("/count/cameras/{camera_id}/pause")
async def camera_pause(camera_id: str, request: Request):
    """특정 카메라를 일시 해제한다 (다른 앱이 웹캠을 사용해야 할 때 호출)."""
    mgr = request.app.state.camera_manager
    try:
        ok = mgr.pause_camera(camera_id, timeout=5.0)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
    if ok:
        return {"camera_id": camera_id, "message": "카메라 해제됨 (paused)", "paused": True}
    return {"camera_id": camera_id, "message": "pause 타임아웃 — 카메라가 여전히 사용 중일 수 있음", "paused": False}


@router.post("/count/cameras/{camera_id}/resume")
async def camera_resume(camera_id: str, request: Request):
    """pause 상태의 특정 카메라를 재개한다."""
    mgr = request.app.state.camera_manager
    try:
        mgr.resume_camera(camera_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
    return {"camera_id": camera_id, "message": "카메라 재개됨 (resumed)"}


# ──────────────────────────────────────────────────────────────
# 하위 호환 엔드포인트 (기존 /count/status 경로 유지)
# ──────────────────────────────────────────────────────────────

@router.get("/count/status")
async def count_status(request: Request, camera_id: str | None = None):
    """카운팅 서비스 상태 반환.

    ``camera_id`` 쿼리 파라미터로 특정 카메라를 지정할 수 있다.
    미지정 시 첫 번째 카메라의 상태를 반환한다.
    """
    mgr = request.app.state.camera_manager
    if camera_id:
        svc = mgr.get(camera_id)
        if svc is None:
            raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
        return svc.get_stats()
    stats = mgr.list_stats()
    return stats[0] if stats else {"message": "no cameras registered"}


@router.get("/count/events")
async def count_events(request: Request, camera_id: str | None = None, limit: int = 100):
    """최근 라인 크로싱 이벤트 목록 반환.

    ``camera_id`` 미지정 시 첫 번째 카메라의 이벤트를 반환한다.
    """
    mgr = request.app.state.camera_manager
    if camera_id is None:
        stats = mgr.list_stats()
        if not stats:
            return []
        camera_id = stats[0]["camera_id"]
    repo = request.app.state.repo
    events = repo.tracking_event.get_events(camera_id, limit=limit)
    return [
        {
            "id": e["id"],
            "camera_id": e["camera_id"],
            "tracker_id": e["tracker_id"],
            "direction": e["direction"],
            "count_change": e["count_change"],
            "current_count": e["current_count"],
            "confidence": e["confidence"],
            "snapshot_path": e["snapshot_path"],
            "created_at": e["created_at"].isoformat() if e["created_at"] else None,
        }
        for e in events
    ]


@router.post("/count/reset")
async def count_reset(request: Request, camera_id: str | None = None):
    """카운트 초기화.

    ``camera_id`` 미지정 시 첫 번째 카메라를 초기화한다.
    """
    mgr = request.app.state.camera_manager
    if camera_id is None:
        stats = mgr.list_stats()
        if not stats:
            return {"message": "no cameras registered"}
        camera_id = stats[0]["camera_id"]
    svc = mgr.get(camera_id)
    if svc is None:
        raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
    svc._current_count = 0
    svc._in_count = 0
    svc._out_count = 0
    logger.info("PersonCounter[%s]: count manually reset to 0", camera_id)
    return {"camera_id": camera_id, "message": "카운트 초기화 완료"}


@router.post("/count/camera/pause")
async def compat_camera_pause(request: Request, camera_id: str | None = None):
    """하위 호환 pause 엔드포인트."""
    mgr = request.app.state.camera_manager
    if camera_id is None:
        stats = mgr.list_stats()
        if not stats:
            return {"message": "no cameras registered", "paused": False}
        camera_id = stats[0]["camera_id"]
    try:
        ok = mgr.pause_camera(camera_id, timeout=5.0)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
    if ok:
        return {"camera_id": camera_id, "message": "카메라 해제됨 (paused)", "paused": True}
    return {"camera_id": camera_id, "message": "pause 타임아웃", "paused": False}


@router.post("/count/camera/resume")
async def compat_camera_resume(request: Request, camera_id: str | None = None):
    """하위 호환 resume 엔드포인트."""
    mgr = request.app.state.camera_manager
    if camera_id is None:
        stats = mgr.list_stats()
        if not stats:
            return {"message": "no cameras registered"}
        camera_id = stats[0]["camera_id"]
    try:
        mgr.resume_camera(camera_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"camera_id={camera_id!r} not found")
    return {"camera_id": camera_id, "message": "카메라 재개됨 (resumed)"}
