import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from server.config import settings
from server.services.common.result_publisher import publish_plant_detection
from server.services.plant.lettuce_service import LETTUCE_CROP_NAME, LETTUCE_CROP_TYPE

router = APIRouter(tags=["lettuce"])
logger = logging.getLogger(__name__)


@router.get("/lettuce/status")
async def lettuce_status(request: Request):
    """상추 탐지 모델 준비 상태 반환."""
    svc = request.app.state.lettuce_service
    return {"ready": svc.is_ready()}


@router.post("/lettuce/detect")
async def lettuce_detect(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.0, le=1.0, description="탐지 신뢰도 임계값"),
):
    """업로드된 이미지에서 상추 병해를 탐지하고 결과를 반환한다.

    결과는 plant_detection_logs 테이블에 crop_type=11(상추)로 저장된다.
    """
    svc = request.app.state.lettuce_service

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"detail": "이미지를 디코딩할 수 없습니다."})

    result = svc.detect(frame, conf_threshold=conf)

    # 병해(비정상) 탐지가 하나라도 있으면 정상(normal) 탐지 결과를 모두 제거
    detections = result.detections
    if any(d.class_name != "normal" for d in detections):
        detections = [d for d in detections if d.class_name != "normal"]
    result.detections = detections

    repo = request.app.state.repo
    camera_id = "cam-lettuce-01"
    _base_url = settings.FASTAPI_PUBLIC_HOST.rstrip("/") if settings.FASTAPI_PUBLIC_HOST else None

    for d in result.detections:
        _x1, _y1, _x2, _y2 = (int(v) for v in d.bbox)
        _crop = frame[max(0, _y1):_y2, max(0, _x1):_x2]
        _img_bytes: bytes | None = None
        if _crop.size > 0:
            _ok, _enc = cv2.imencode(".jpg", _crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if _ok:
                _img_bytes = _enc.tobytes()

        try:
            _row = repo.plant_detection_log.create(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=d.bbox,
                disease_code=d.disease_code,
                disease_label=d.disease_label,
                source="api",
                crop_type=LETTUCE_CROP_TYPE,
                crop_name=LETTUCE_CROP_NAME,
                image_data=_img_bytes,
            )
            _img_url = f"{_base_url}/api/lettuce/logs/{_row.id}/image" if _row and _img_bytes and _base_url else None
        except Exception as e:
            logger.warning("lettuce detection log save failed: %s", e)
            _img_url = None

        publish_plant_detection(
            source=camera_id,
            class_name=d.class_name,
            disease_code=d.disease_code,
            disease_label=d.disease_label,
            confidence=d.confidence,
            bbox=d.bbox,
            crop_type=LETTUCE_CROP_TYPE,
            crop_name=LETTUCE_CROP_NAME,
            shooting_type=None,
            grow_stage=None,
            area=None,
            image_url=_img_url,
        )

    return {
        "count": len(result),
        "meta": {
            "crop_type": LETTUCE_CROP_TYPE,
            "crop_name": LETTUCE_CROP_NAME,
        },
        "detections": [
            {
                "class_name": d.class_name,
                "disease_code": d.disease_code,
                "disease_label": d.disease_label,
                "confidence": round(d.confidence, 4),
                "bbox": [round(v, 2) for v in d.bbox],
            }
            for d in result.detections
        ],
    }


@router.get("/lettuce/logs/{log_id}/image")
async def lettuce_log_image(log_id: int, request: Request):
    """저장된 상추 탐지 이미지 다운로드."""
    repo = request.app.state.repo
    row = repo.plant_detection_log.get(log_id)
    if row is None or not row.image_data:
        return JSONResponse(status_code=404, content={"detail": "이미지 없음"})
    return Response(
        content=row.image_data,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=lettuce_{log_id}.jpg"},
    )
