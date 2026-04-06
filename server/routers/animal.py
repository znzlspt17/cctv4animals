import logging

import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, Query
from fastapi.responses import JSONResponse

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
    camera_id = request.headers.get("X-Camera-Id", "unknown")
    for d in result.detections:
        try:
            repo.animal_detection_log.create(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=d.bbox,
                source="api",
            )
        except Exception as e:
            logger.warning("animal detection log save failed: %s", e)

        publish_animal_detection(
            source=camera_id,
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=d.bbox,
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


@router.post("/animal/reset")
async def animal_reset(request: Request):
    """라인 추적 상태 초기화 (카메라 전환 등)."""
    svc = request.app.state.animal_service
    svc.reset_tracker()
    return {"message": "Animal tracker reset complete"}
