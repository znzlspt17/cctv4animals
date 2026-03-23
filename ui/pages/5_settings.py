"""설정 페이지."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import requests
import streamlit as st

from ui.components.sidebar import render_sidebar

st.set_page_config(page_title="설정 — DeepFace Live", layout="wide")
render_sidebar()

st.title("⚙️ 설정")

API_BASE = st.session_state.get("api_base_url", "http://localhost:8000/api")

st.warning(
    "⚠️ 변경된 설정은 현재 세션에만 적용됩니다. 서버 재시작 시 .env 기본값으로 복원됩니다."
)

# ── API 서버 연결 ──
st.subheader("🔌 API 서버")
new_api_base = st.text_input(
    "API Base URL",
    value=st.session_state.get("api_base_url", "http://localhost:8000/api"),
)
if new_api_base != st.session_state.get("api_base_url"):
    st.session_state["api_base_url"] = new_api_base

# DB 연결 상태
col_db1, col_db2 = st.columns(2)
with col_db1:
    if st.button("🔍 DB 연결 상태 확인"):
        try:
            resp = requests.get(f"{API_BASE}/persons", timeout=5)
            if resp.status_code == 200:
                st.success("✅ DB 연결 정상 (persons 테이블 접근 가능)")
            else:
                st.error(f"❌ DB 오류 (HTTP {resp.status_code})")
        except requests.ConnectionError:
            st.error("❌ API 서버에 연결할 수 없습니다.")
        except Exception as e:
            st.error(f"❌ 오류: {e}")

with col_db2:
    st.markdown("**현재 DB 백엔드**: 서버 설정에 따름 (MySQL / Redis)")

st.divider()

# ── 서버 설정 (읽기 전용) ──
st.subheader("🖥️ 서버 설정 (읽기 전용 — 변경 시 서버 재시작 필요)")
st.info(
    "아래 설정은 서버의 .env 파일에서 관리됩니다. "
    "변경하려면 .env 파일을 수정하고 서버를 재시작하세요."
)

col_sv1, col_sv2, col_sv3 = st.columns(3)
with col_sv1:
    st.text_input("DeepFace 모델", value="VGG-Face", disabled=True)
with col_sv2:
    st.text_input("얼굴 감지기", value="retinaface", disabled=True)
with col_sv3:
    st.text_input("거리 메트릭", value="cosine", disabled=True)

st.divider()

# ── 등록 조건 슬라이더 ──
st.subheader("📸 등록 조건")
st.caption("얼굴 등록 시 적용되는 최소 품질 조건입니다.")

col_reg1, col_reg2, col_reg3 = st.columns(3)
with col_reg1:
    st.session_state["face_min_confidence"] = st.slider(
        "최소 감지 신뢰도 (FACE_MIN_CONFIDENCE)",
        min_value=0.50,
        max_value=1.00,
        step=0.05,
        value=st.session_state.get("face_min_confidence", 0.90),
    )
with col_reg2:
    st.session_state["face_min_size"] = st.slider(
        "최소 얼굴 크기 (FACE_MIN_SIZE)",
        min_value=56,
        max_value=224,
        step=8,
        value=st.session_state.get("face_min_size", 112),
    )
with col_reg3:
    st.session_state["face_blur_threshold"] = st.slider(
        "블러 임계값 (FACE_BLUR_THRESHOLD)",
        min_value=10.0,
        max_value=500.0,
        step=10.0,
        value=st.session_state.get("face_blur_threshold", 100.0),
    )

st.divider()

# ── 검출 조건 슬라이더 ──
st.subheader("🔍 실시간 검출 조건")
st.caption("실시간 인식 시 적용되는 조건입니다.")

col_det1, col_det2 = st.columns(2)
with col_det1:
    st.session_state["face_min_confidence_rt"] = st.slider(
        "실시간 최소 감지 신뢰도",
        min_value=0.50,
        max_value=1.00,
        step=0.05,
        value=st.session_state.get("face_min_confidence_rt", 0.80),
    )
with col_det2:
    st.session_state["face_min_size_rt"] = st.slider(
        "실시간 최소 얼굴 크기",
        min_value=28,
        max_value=112,
        step=4,
        value=st.session_state.get("face_min_size_rt", 56),
    )

col_det3, col_det4 = st.columns(2)
with col_det3:
    st.session_state["frame_skip"] = st.slider(
        "프레임 스킵 (RECOGNITION_FRAME_SKIP)",
        min_value=1,
        max_value=10,
        step=1,
        value=st.session_state.get("frame_skip", 3),
        key="settings_frame_skip",
    )
with col_det4:
    st.session_state["log_dedup_seconds"] = st.slider(
        "로그 중복 억제 (초)",
        min_value=0,
        max_value=60,
        step=5,
        value=st.session_state.get("log_dedup_seconds", 10),
    )

st.divider()

# ── 공통 슬라이더 ──
st.subheader("🎯 공통 인식 설정")

st.session_state["recognition_threshold"] = st.slider(
    "인식 임계값 (RECOGNITION_THRESHOLD) — 낮을수록 엄격",
    min_value=0.20,
    max_value=0.80,
    step=0.05,
    value=st.session_state.get("recognition_threshold", 0.40),
)

st.divider()

# ── 오버레이 표시 옵션 ──
st.subheader("🏷️ 오버레이 표시 옵션")
st.caption("실시간 인식 화면에서 표시할 정보를 선택합니다.")

col_ov1, col_ov2, col_ov3 = st.columns(3)
with col_ov1:
    st.session_state["overlay_show_name"] = st.checkbox(
        "이름 표시",
        value=st.session_state.get("overlay_show_name", True),
        key="settings_overlay_name",
    )
with col_ov2:
    st.session_state["overlay_show_phone"] = st.checkbox(
        "전화번호 표시",
        value=st.session_state.get("overlay_show_phone", False),
        key="settings_overlay_phone",
    )
with col_ov3:
    st.session_state["overlay_show_address"] = st.checkbox(
        "주소 표시",
        value=st.session_state.get("overlay_show_address", False),
        key="settings_overlay_address",
    )

st.session_state["display_mode"] = st.selectbox(
    "표시 모드",
    options=["name", "name+phone", "name+address", "all"],
    index=["name", "name+phone", "name+address", "all"].index(
        st.session_state.get("display_mode", "name")
    ),
    key="settings_display_mode",
)

st.divider()

# ── 현재 세션 설정 요약 ──
st.subheader("📝 현재 세션 설정 요약")

summary_keys = [
    ("API Base URL", "api_base_url"),
    ("인식 임계값", "recognition_threshold"),
    ("등록 최소 신뢰도", "face_min_confidence"),
    ("등록 최소 얼굴 크기", "face_min_size"),
    ("등록 블러 임계값", "face_blur_threshold"),
    ("실시간 최소 신뢰도", "face_min_confidence_rt"),
    ("실시간 최소 얼굴 크기", "face_min_size_rt"),
    ("프레임 스킵", "frame_skip"),
    ("로그 중복 억제 (초)", "log_dedup_seconds"),
    ("표시 모드", "display_mode"),
]

summary_data = []
for label, key in summary_keys:
    summary_data.append({"설정": label, "값": str(st.session_state.get(key, "-"))})

st.dataframe(summary_data, use_container_width=True)
