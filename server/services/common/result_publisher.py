"""추론 결과 외부 전송 모듈.

각 카메라/서비스에서 이벤트가 발생할 때 호출하면
설정된 외부 서버로 JSON 데이터를 POST 전송한다.
전송 실패 시 로그만 남기고 예외를 전파하지 않는다.
"""

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from server.config import settings

logger = logging.getLogger(__name__)


def _post(path: str, body: dict[str, Any]) -> None:
    """내부 공통 POST 헬퍼. 실패 시 예외를 전파하지 않는다."""
    base = settings.RESULT_PUBLISHER_BASE_URL
    if not base:
        return

    url = base.rstrip("/") + path
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=settings.RESULT_PUBLISHER_TIMEOUT) as resp:
            status = resp.status
            if status >= 400:
                logger.warning("Result publisher: server returned %d for %s", status, path)
            else:
                logger.debug("Result publisher: sent %s → %d", path, status)
    except urllib.error.URLError as e:
        logger.warning("Result publisher: connection failed (%s): %s", path, e)
    except TimeoutError:
        logger.warning("Result publisher: timeout (%s)", path)
    except Exception as e:
        logger.warning("Result publisher: unexpected error (%s): %s", path, e)


def publish_animal_detection(
    source: str,
    class_name: str,
    confidence: float,
    bbox: list[float],
    image_url: str | None = None,
) -> None:
    """동물 탐지 이벤트를 POST /api/detections/animal 으로 전송한다."""
    x1, y1, x2, y2 = (bbox + [0, 0, 0, 0])[:4]
    body: dict[str, Any] = {
        "source": source,
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "bbox_x1": round(x1, 2),
        "bbox_y1": round(y1, 2),
        "bbox_x2": round(x2, 2),
        "bbox_y2": round(y2, 2),
        "detected_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if image_url:
        body["image_url"] = image_url
    _post("/api/detections/animal", body)


def publish_plant_detection(
    source: str,
    class_name: str,
    disease_code: int | None,
    disease_label: str | None,
    confidence: float,
    bbox: list[float],
    crop_type: int | None,
    crop_name: str | None,
    shooting_type: int | None,
    grow_stage: int | None,
    area: int | None,
    image_url: str | None = None,
) -> None:
    """식물 탐지 이벤트를 POST /api/detections/plant 으로 전송한다."""
    x1, y1, x2, y2 = (bbox + [0, 0, 0, 0])[:4]
    body: dict[str, Any] = {
        "source": source,
        "class_name": class_name,
        "disease_code": disease_code,
        "disease_label": disease_label,
        "confidence": round(confidence, 4),
        "bbox_x1": round(x1, 2),
        "bbox_y1": round(y1, 2),
        "bbox_x2": round(x2, 2),
        "bbox_y2": round(y2, 2),
        "crop_type": str(crop_type) if crop_type is not None else None,
        "crop_name": crop_name,
        "shooting_type": shooting_type,
        "grow_stage": grow_stage,
        "area": area,
        "detected_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if image_url:
        body["image_url"] = image_url
    _post("/api/detections/plant", body)


def publish_event(
    event_type: str,
    camera_id: str,
    payload: dict[str, Any],
) -> None:
    """범용 이벤트를 /api/events 로 POST 전송한다 (person, face 등)."""
    body: dict[str, Any] = {
        "event_type": event_type,
        "camera_id": camera_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "payload": payload,
    }
    _post("/api/events", body)
