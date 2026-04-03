"""바운딩 박스 + 텍스트 오버레이 렌더러."""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 한글 지원 폰트 탐색 (Windows 우선, Linux/Mac 대체)
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",       # Windows 맑은 고딕
    "C:/Windows/Fonts/malgunbd.ttf",     # Windows 맑은 고딕 Bold
    "C:/Windows/Fonts/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
_FONT_PATH: str | None = next(
    (p for p in _FONT_CANDIDATES if os.path.exists(p)), None
)


def _load_pil_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _put_text_pil(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_size: int = 20,
    text_color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 0, 0),
    bg_alpha: float = 0.6,
) -> np.ndarray:
    """PIL로 한글 텍스트를 프레임에 렌더링한다."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = _load_pil_font(font_size)

    tx, ty = pos
    bbox = draw.textbbox((tx, ty), text, font=font)
    pad = 4
    bg_box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)

    # 반투명 배경
    overlay_pil = img_pil.copy()
    overlay_draw = ImageDraw.Draw(overlay_pil)
    overlay_draw.rectangle(bg_box, fill=(0, 0, 0))
    img_pil = Image.blend(img_pil, overlay_pil, alpha=bg_alpha)

    draw = ImageDraw.Draw(img_pil)
    draw.text((tx, ty), text, font=font, fill=text_color[::-1])  # RGB

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


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

        # 텍스트 렌더링 (PIL — 한글 지원)
        font_size = 20
        line_height = font_size + 8

        for i, line_text in enumerate(lines):
            text_y = y - 10 - (len(lines) - 1 - i) * line_height
            if text_y < font_size:
                text_y = y + h + 8 + i * line_height

            frame = _put_text_pil(frame, line_text, (x, text_y), font_size=font_size)

    return frame
