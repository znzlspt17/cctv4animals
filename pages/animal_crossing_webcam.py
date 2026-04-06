"""동물 실시간 탐지 — 웹캠 페이지."""

import sys
sys.path.insert(0, ".")

import queue
import time

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

st.set_page_config(page_title="동물 실시간 탐지 (웹캠)", layout="wide")
st.title("🐾 동물 실시간 탐지 — 웹캠")
st.caption("웹캠 영상에서 동물을 실시간으로 탐지합니다.")

_API_BASE = "http://localhost:8000"


def _api_pause() -> bool:
    try:
        r = requests.post(f"{_API_BASE}/api/count/camera/pause", timeout=7)
        return r.json().get("paused", False)
    except Exception:
        return False


def _api_resume() -> None:
    try:
        requests.post(f"{_API_BASE}/api/count/camera/resume", timeout=3)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────
# 서비스 로드 (캐시)
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_animal_service():
    from server.services.animal.animal_service import AnimalService
    svc = AnimalService()
    svc.warmup()
    return svc


# ──────────────────────────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────────────────────────
if "det_count" not in st.session_state:
    st.session_state["det_count"] = 0
if "det_log" not in st.session_state:
    st.session_state["det_log"] = []
if "det_queue" not in st.session_state:
    st.session_state["det_queue"] = queue.Queue()
if "was_playing" not in st.session_state:
    st.session_state["was_playing"] = False
if "cam_released" not in st.session_state:
    st.session_state["cam_released"] = False

# ──────────────────────────────────────────────────────────────────
# 페이지 진입 시 자동으로 PersonCounter 카메라 해제 (한 번만)
# ──────────────────────────────────────────────────────────────────
if not st.session_state["cam_released"]:
    with st.spinner("📷 PersonCounter 카메라 해제 중…"):
        ok = _api_pause()
    if ok:
        st.session_state["cam_released"] = True
        st.success("✅ 카메라 사용 가능 — 아래 START 버튼을 누르세요.")
    else:
        st.warning("⚠️ PersonCounter API 연결 실패. FastAPI 서버가 실행 중인지 확인하세요.")

# ──────────────────────────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    conf_thr = st.slider("Confidence 임계값", 0.1, 1.0, 0.3, 0.05, key="wc_conf")

    st.divider()
    if st.button("🔄 카운터 리셋", use_container_width=True):
        st.session_state["det_count"] = 0
        st.session_state["det_log"] = []
        st.success("리셋 완료")

    st.divider()
    st.subheader("📷 카메라 수동 제어")
    col_p, col_r = st.columns(2)
    with col_p:
        if st.button("⏸ 해제", use_container_width=True):
            ok = _api_pause()
            st.session_state["cam_released"] = ok
            st.toast("카메라 해제됨" if ok else "해제 실패", icon="📷")
    with col_r:
        if st.button("▶ 재개", use_container_width=True):
            _api_resume()
            st.session_state["cam_released"] = False
            st.toast("PersonCounter 카메라 재개됨", icon="📷")

    cam_status = "🟢 해제됨" if st.session_state["cam_released"] else "🔴 점유 중"
    st.caption(f"카메라 상태: {cam_status}")


# ──────────────────────────────────────────────────────────────────
# VideoProcessor 팩토리
# recv()는 WebRTC 내부 스레드에서 실행 → st.session_state 접근 불가.
# 메인 스레드에서 필요한 값을 미리 캡처해 클로저로 전달한다.
# ──────────────────────────────────────────────────────────────────
def _make_processor_factory(dq, svc, conf):
    class _Processor(VideoProcessorBase):
        def __init__(self):
            self._svc  = svc
            self._dq   = dq
            self._conf = conf

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")

            try:
                result = self._svc.detect(img_bgr, conf_threshold=self._conf)
            except Exception:
                result = None

            if result:
                for d in result.detections:
                    bx1, by1, bx2, by2 = int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])
                    cv2.rectangle(img_bgr, (bx1, by1), (bx2, by2), (0, 200, 0), 2)
                    cv2.putText(img_bgr, f"{d.class_name} {d.confidence:.2f}",
                                (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                    self._dq.put({
                        "시각": time.strftime("%H:%M:%S"),
                        "클래스": d.class_name,
                        "confidence": round(d.confidence, 4),
                    })

            img_rgb = np.ascontiguousarray(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
            )
            return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")

    return _Processor


# ──────────────────────────────────────────────────────────────────
# 메인 레이아웃
# ──────────────────────────────────────────────────────────────────
col_video, col_panel = st.columns([3, 1])

# 메인 스레드에서 값 캡처 (WebRTC 스레드에 안전하게 전달)
_dq_captured   = st.session_state["det_queue"]
_svc_captured  = load_animal_service()
_conf_captured = float(st.session_state.get("wc_conf", 0.3))

with col_video:
    webrtc_ctx = webrtc_streamer(
        key="animal-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=_make_processor_factory(
            _dq_captured, _svc_captured, _conf_captured,
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    # 스트림 종료 감지 → PersonCounter 카메라 자동 재개
    currently_playing = webrtc_ctx.state.playing
    if st.session_state["was_playing"] and not currently_playing:
        _api_resume()
        st.session_state["cam_released"] = False
        st.session_state["was_playing"] = False
        st.info("📷 PersonCounter 카메라 자동 재개됨")
    if currently_playing:
        st.session_state["was_playing"] = True

with col_panel:
    st.subheader("📊 탐지 카운터")
    metric_total = st.empty()
    st.divider()
    st.subheader("📋 탐지 로그 (최근 30건)")
    log_placeholder = st.empty()


# ──────────────────────────────────────────────────────────────────
# 메인 루프 — 큐 → UI 갱신
# ──────────────────────────────────────────────────────────────────
def _render_panel():
    metric_total.metric("총 탐지 수", st.session_state["det_count"])
    logs = st.session_state["det_log"]
    if logs:
        log_placeholder.dataframe(logs, use_container_width=True)
    else:
        log_placeholder.info("탐지된 동물 없음")


if webrtc_ctx.state.playing:
    dq = st.session_state["det_queue"]
    while True:
        try:
            ev = dq.get(timeout=0.1)
            st.session_state["det_count"] += 1
            logs = st.session_state["det_log"]
            logs.insert(0, ev)
            st.session_state["det_log"] = logs[:30]
        except queue.Empty:
            pass
        _render_panel()
        time.sleep(0.2)
else:
    _render_panel()
