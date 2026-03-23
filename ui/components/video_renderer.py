"""바운딩 박스 + 텍스트 오버레이 렌더러."""

import cv2
import numpy as np


def draw_results(
    frame: np.ndarray,
    results: list[dict],
    overlay_options: dict | None = None,
) -> np.ndarray:
    """인식 결과를 프레임에 오버레이한다.

    Args:
        frame: BGR numpy 이미지
        results: RecognitionResult dict 리스트
            각 항목: person_id, person_name, display_name, confidence, bbox, alerts
        overlay_options: {"show_name": bool, "show_phone": bool, "show_address": bool}

    Returns:
        오버레이가 그려진 프레임 (원본 수정)
    """
    if overlay_options is None:
        overlay_options = {
            "show_name": True,
            "show_phone": False,
            "show_address": False,
        }

    for r in results:
        bbox = r.get("bbox", [])
        if len(bbox) < 4:
            continue

        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        person_id = r.get("person_id")
        person_name = r.get("person_name", "")
        display_name = r.get("display_name", "")
        confidence = r.get("confidence", 0.0)
        alerts = r.get("alerts", [])
        conf_pct = f"{confidence * 100:.0f}%"

        # 색상 결정: 알림 → 주황, 매칭 → 초록, Unknown → 빨강
        if alerts:
            color = (0, 165, 255)  # BGR 주황
        elif person_id is not None:
            color = (0, 255, 0)  # BGR 초록
        else:
            color = (0, 0, 255)  # BGR 빨강

        # 바운딩 박스
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # 텍스트 라인 구성
        lines = []
        if person_id is not None:
            label = display_name or person_name or f"ID:{person_id}"
            if overlay_options.get("show_name", True):
                lines.append(f"{label} ({conf_pct})")
            else:
                lines.append(conf_pct)

            # 추가 정보 (phone, address 등은 result에 포함되어 있을 수 있음)
            if overlay_options.get("show_phone") and r.get("phone"):
                lines.append(r["phone"])
            if overlay_options.get("show_address") and r.get("address"):
                lines.append(r["address"])
        else:
            lines.append(f"Unknown ({conf_pct})")

        # 텍스트 렌더링 (반투명 배경 + 흰색 글씨)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_height = 28

        for i, line_text in enumerate(lines):
            text_y = y - 10 - (len(lines) - 1 - i) * line_height
            if text_y < 15:
                text_y = y + h + 20 + i * line_height

            (tw, th), _ = cv2.getTextSize(line_text, font, font_scale, thickness)
            # 반투명 배경
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x, text_y - th - 4),
                (x + tw + 6, text_y + 4),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            # 텍스트
            cv2.putText(
                frame,
                line_text,
                (x + 3, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

    return frame
