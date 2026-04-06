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


def publish_event(
    event_type: str,
    camera_id: str,
    payload: dict[str, Any],
) -> None:
    """이벤트를 외부 서버로 POST 전송한다.

    Args:
        event_type: 이벤트 종류 식별자
                    (예: "person_crossing", "animal_detection",
                         "face_recognition", "plant_detection")
        camera_id:  이벤트가 발생한 카메라/소스 식별자.
        payload:    이벤트 세부 데이터 딕셔너리.
    """
    if not settings.RESULT_PUBLISHER_URL:
        return

    body: dict[str, Any] = {
        "event_type": event_type,
        "camera_id": camera_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "payload": payload,
    }

    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url=settings.RESULT_PUBLISHER_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=settings.RESULT_PUBLISHER_TIMEOUT) as resp:
            status = resp.status
            if status >= 400:
                logger.warning(
                    "Result publisher: server returned %d for event_type=%s camera_id=%s",
                    status, event_type, camera_id,
                )
            else:
                logger.debug(
                    "Result publisher: sent event_type=%s camera_id=%s → %d",
                    event_type, camera_id, status,
                )
    except urllib.error.URLError as e:
        logger.warning(
            "Result publisher: connection failed (event_type=%s camera_id=%s): %s",
            event_type, camera_id, e,
        )
    except TimeoutError:
        logger.warning(
            "Result publisher: timeout (event_type=%s camera_id=%s)",
            event_type, camera_id,
        )
    except Exception as e:
        logger.warning(
            "Result publisher: unexpected error (event_type=%s camera_id=%s): %s",
            event_type, camera_id, e,
        )
