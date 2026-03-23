"""DeepFace Live — Streamlit 멀티페이지 앱 엔트리포인트."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="DeepFace Live",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session_state 기본값 초기화 ──
_defaults = {
    "api_base_url": "http://localhost:8000/api",
    "display_mode": "name",
    "frame_skip": 3,
    "recognition_active": True,
    "recognition_threshold": 0.40,
    "face_min_confidence": 0.90,
    "face_min_size": 112,
    "face_blur_threshold": 100.0,
    "face_min_confidence_rt": 0.80,
    "face_min_size_rt": 56,
    "log_dedup_seconds": 10,
    "overlay_show_name": True,
    "overlay_show_phone": False,
    "overlay_show_address": False,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── 사이드바 렌더링 ──
from ui.components.sidebar import render_sidebar  # noqa: E402

render_sidebar()

# ── 메인 ──
st.title("🎭 DeepFace Live")
st.markdown(
    """
    왼쪽 사이드바에서 페이지를 선택하세요.

    | 페이지 | 설명 |
    |--------|------|
    | **실시간 인식** | 웹캠으로 실시간 얼굴 인식 |
    | **얼굴 등록** | 새로운 얼굴 등록 |
    | **인물 관리** | 등록된 인물 관리 |
    | **로그** | 인식 로그 조회 |
    | **설정** | 시스템 설정 |
    """
)
