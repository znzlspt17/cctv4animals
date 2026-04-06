import logging

import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, Query
from fastapi.responses import JSONResponse

router = APIRouter(tags=["plant"])
logger = logging.getLogger(__name__)


@router.get("/plant/status")
async def plant_status(request: Request):
    """식물 탐지 모델 준비 상태 반환."""
    svc = request.app.state.plant_service
    return {"ready": svc.is_ready()}


@router.post("/plant/detect")
async def plant_detect(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(default=0.4, ge=0.0, le=1.0, description="탐지 신뢰도 임계값"),
    crop_type: int | None = Query(default=None, description="작물 명칭 코드 (0=딸기, 1=토마토, 2=파프리카, 3=오이, 4=딸기시설, 5=고추, 6=수박, 7=멜론, 8=호박, 9=가지, 10=배추, 11=상추)"),
    shooting_type: int | None = Query(default=None, description="촬영 유형 코드 (0=Normal, 1=Complaint, 2=Insect, 3=Damage)"),
    grow_stage: int | None = Query(default=None, description="작물 생육 단계 코드 (11=육묘기, 12=생장기, 13=착화/과실기)"),
    area: int | None = Query(default=None, description="촬영 부위 코드 (0=None, 1=Fruit, 2=Flower, 3=Leaf, 4=Branch, 5=Stem, 6=Root)"),
):
    """업로드된 이미지에서 식물을 탐지하고 결과를 반환한다."""
    svc = request.app.state.plant_service

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"detail": "이미지를 디코딩할 수 없습니다."})

    result = svc.detect(
        frame,
        conf_threshold=conf,
        crop_type=crop_type,
        shooting_type=shooting_type,
        grow_stage=grow_stage,
        area=area,
    )

    # 탐지 결과 DB 저장
    repo = request.app.state.repo
    for d in result.detections:
        try:
            repo.plant_detection_log.create(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=d.bbox,
                disease_code=d.disease_code,
                disease_label=d.disease_label,
                source="api",
                crop_type=result.crop_type,
                crop_name=result.crop_name,
                shooting_type=result.shooting_type,
                shooting_type_name=result.shooting_type_name,
                grow_stage=result.grow_stage,
                grow_stage_name=result.grow_stage_name,
                area=result.area,
                area_name=result.area_name,
            )
        except Exception as e:
            logger.warning("plant detection log save failed: %s", e)

    return {
        "count": len(result),
        "meta": {
            "crop_type": result.crop_type,
            "crop_name": result.crop_name,
            "shooting_type": result.shooting_type,
            "shooting_type_name": result.shooting_type_name,
            "grow_stage": result.grow_stage,
            "grow_stage_name": result.grow_stage_name,
            "area": result.area,
            "area_name": result.area_name,
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
