"""실시간 얼굴 인식 페이지."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from ui.components.sidebar import render_sidebar
from ui.components.video_renderer import draw_results

st.set_page_config(page_title="실시간 인식 — DeepFace Live", layout="wide")
render_sidebar()

st.title("📹 실시간 얼굴 인식")

API_BASE = st.session_state.get("api_base_url", "http://localhost:8000/api")

# ── 컨트롤 패널 ──
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    st.session_state["recognition_active"] = st.toggle(
        "인식 활성화", value=st.session_state.get("recognition_active", True)
    )
with col_ctrl2:
    st.session_state["frame_skip"] = st.slider(
        "프레임 스킵 (N프레임마다 1회 인식)",
        min_value=1,
        max_value=10,
        value=st.session_state.get("frame_skip", 3),
    )
with col_ctrl3:
    st.session_state["display_mode"] = st.selectbox(
        "표시 모드",
        options=["name", "name+phone", "name+address", "all"],
        index=["name", "name+phone", "name+address", "all"].index(
            st.session_state.get("display_mode", "name")
        ),
    )


class FaceRecognitionProcessor(VideoProcessorBase):
    """streamlit-webrtc VideoProcessor: N프레임마다 /api/recognize 호출."""

    def __init__(self):
        self._frame_count = 0
        self._last_results: list[dict] = []
        # 메인 스레드에서 업데이트되는 설정값 (session_state 대신 인스턴스 속성 사용)
        self.recognition_active: bool = True
        self.frame_skip: int = 3
        self.display_mode: str = "name"
        self.api_base: str = "http://localhost:9000/api"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.recognition_active:
            self._frame_count += 1
            if self._frame_count % self.frame_skip == 0:
                self._last_results = self._call_recognize(img)

        # 오버레이 옵션
        overlay_opts = {
            "show_name": True,
            "show_phone": self.display_mode in ("name+phone", "all"),
            "show_address": self.display_mode in ("name+address", "all"),
        }

        # 결과 오버레이
        if self._last_results:
            img = draw_results(img, self._last_results, overlay_opts)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _call_recognize(self, img: np.ndarray) -> list[dict]:
        """프레임을 JPEG로 인코딩 후 /api/recognize 호출."""
        try:
            _, buffer = cv2.imencode(".jpg", img)
            resp = requests.post(
                f"{self.api_base}/recognize",
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])
        except Exception:
            pass
        return []


# ── WebRTC 스트리머 ──
ctx = webrtc_streamer(
    key="face-recognition",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceRecognitionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 메인 스레드에서 processor 속성 업데이트 (session_state → 인스턴스 속성)
if ctx.video_processor:
    ctx.video_processor.recognition_active = st.session_state.get("recognition_active", True)
    ctx.video_processor.frame_skip = st.session_state.get("frame_skip", 3)
    ctx.video_processor.display_mode = st.session_state.get("display_mode", "name")
    ctx.video_processor.api_base = st.session_state.get("api_base_url", API_BASE)

st.divider()

# ── 최근 인식 결과 표시 ──
st.subheader("📋 최근 인식 결과")

if st.button("🔄 최근 로그 새로고침"):
    pass  # 아래에서 항상 로드

try:
    resp = requests.get(
        f"{API_BASE}/logs",
        params={"page": 1, "page_size": 10},
        timeout=5,
    )
    if resp.status_code == 200:
        logs = resp.json()
        if logs:
            log_data = []
            for log in logs:
                log_data.append(
                    {
                        "시간": log.get("recognized_at", ""),
                        "인물": log.get("person_name", "Unknown"),
                        "신뢰도": f"{log.get('confidence', 0) * 100:.1f}%",
                    }
                )
            st.dataframe(log_data, use_container_width=True)
        else:
            st.info("아직 인식 기록이 없습니다.")
    else:
        st.warning("로그를 불러올 수 없습니다.")
except requests.ConnectionError:
    st.error("API 서버에 연결할 수 없습니다.")
except Exception as e:
    st.error(f"오류: {e}")

# ── 알림 처리 (페이지 리로드 시 session_state에서 확인) ──
pending_alerts = st.session_state.pop("_pending_alerts", [])
for alert in pending_alerts:
    st.toast(alert.get("message", "알림"), icon="🚨")
