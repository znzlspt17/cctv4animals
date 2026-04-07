import base64
import logging
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse, Response

from server.config import settings
from server.services.common.result_publisher import publish_animal_detection

router = APIRouter(tags=["animal"])
logger = logging.getLogger(__name__)


@router.get("/animal/status")
async def animal_status(request: Request):
    """동물 탐지 모델 준비 상태 반환."""
    svc = request.app.state.animal_service
    return {"ready": svc.is_ready()}


@router.post("/animal/detect")
async def animal_detect(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(default=0.4, ge=0.0, le=1.0),
):
    """업로드된 이미지에서 동물을 탐지하고 결과를 반환한다."""
    svc = request.app.state.animal_service
    repo = request.app.state.repo

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"detail": "이미지를 디코딩할 수 없습니다."})

    result = svc.detect(frame, conf_threshold=conf)

    # 탐지 결과 DB 저장 및 외부 전송
    camera_id = "cam-center-01"
    # FASTAPI_PUBLIC_HOST 미설정 시 이미지 URL 생성 안 함 (localhost를 외부 서버에 보내지 않기 위해)
    _base_url = settings.FASTAPI_PUBLIC_HOST.rstrip("/") if settings.FASTAPI_PUBLIC_HOST else None
    for d in result.detections:
        # bbox 영역 크롭 → JPEG 바이트
        _x1, _y1, _x2, _y2 = (int(v) for v in d.bbox)
        _crop = frame[max(0, _y1):_y2, max(0, _x1):_x2]
        _img_bytes: bytes | None = None
        if _crop.size > 0:
            _ok, _enc = cv2.imencode(".jpg", _crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if _ok:
                _img_bytes = _enc.tobytes()

        try:
            _row = repo.animal_detection_log.create(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=d.bbox,
                source=camera_id,
                image_data=_img_bytes,
            )
            _img_url = f"{_base_url}/api/animal/logs/{_row.id}/image" if _row and _img_bytes and _base_url else None
        except Exception as e:
            logger.error("animal detection log save failed: %s", e, exc_info=True)
            _img_url = None

        publish_animal_detection(
            source=camera_id,
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=d.bbox,
            image_url=_img_url,
        )

    return {
        "count": len(result),
        "detections": [
            {
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": [round(v, 2) for v in d.bbox],
            }
            for d in result.detections
        ],
    }


@router.get("/animal/logs")
async def animal_logs(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
):
    """동물 탐지 로그 조회."""
    repo = request.app.state.repo
    rows = repo.animal_detection_log.query(limit=limit)
    return [
        {
            "id": r.id,
            "source": r.source,
            "class_name": r.class_name,
            "confidence": round(r.confidence, 4),
            "bbox": [r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2],
            "detected_at": r.detected_at,
        }
        for r in rows
    ]


@router.post("/animal/logs/batch")
async def animal_logs_batch(
    request: Request,
    detections: list[dict[str, Any]] = Body(...),
    source: str = Query(default="cam-center-01"),
):
    """탐지 결과 배치 저장. tab6 동영상 처리 등에서 사용.

    Body: [{"class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2],
            "image_b64": str(optional, base64-encoded JPEG)}, ...]
    반환: {"saved": int, "total": int, "ids": [int|null, ...]}
    """
    repo = request.app.state.repo
    saved = 0
    ids: list[int | None] = []
    for d in detections:
        try:
            _img_data: bytes | None = None
            if d.get("image_b64"):
                _img_data = base64.b64decode(d["image_b64"])
            row = repo.animal_detection_log.create(
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=d.get("bbox", []),
                source=source,
                image_data=_img_data,
            )
            ids.append(row.id)
            saved += 1
        except Exception as e:
            logger.error("batch save failed for %s: %s", d, e)
            ids.append(None)
    return {"saved": saved, "total": len(detections), "ids": ids}


@router.get("/animal/logs/{log_id}/image")
async def animal_log_image(log_id: int, request: Request):
    """저장된 동물 탐지 이미지 다운로드."""
    repo = request.app.state.repo
    row = repo.animal_detection_log.get(log_id)
    if row is None or not row.image_data:
        return JSONResponse(status_code=404, content={"detail": "이미지 없음"})
    return Response(
        content=row.image_data,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=animal_{log_id}.jpg"},
    )


@router.post("/animal/reset")
async def animal_reset(request: Request):
    """라인 추적 상태 초기화 (카메라 전환 등)."""
    svc = request.app.state.animal_service
    svc.reset_tracker()
    return {"message": "Animal tracker reset complete"}
