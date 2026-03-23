"""공통 사이드바 컴포넌트."""

import requests
import streamlit as st


def render_sidebar():
    """사이드바: API 연결 상태, 등록 인원 수, 오버레이 옵션."""
    api_base = st.session_state.get("api_base_url", "http://localhost:8000/api")

    with st.sidebar:
        st.header("DeepFace Live")
        st.divider()

        # ── API 서버 연결 상태 ──
        st.subheader("🔌 서버 상태")
        try:
            # FastAPI 기본 docs 엔드포인트로 health check
            resp = requests.get(api_base.replace("/api", "/docs"), timeout=3)
            if resp.status_code == 200:
                st.success("API 서버 연결됨")
            else:
                st.warning(f"서버 응답 이상 (HTTP {resp.status_code})")
        except requests.ConnectionError:
            st.error("API 서버에 연결할 수 없습니다")
        except Exception as e:
            st.error(f"연결 오류: {e}")

        st.divider()

        # ── 등록된 인원 수 ──
        st.subheader("👥 등록 현황")
        try:
            resp = requests.get(f"{api_base}/persons", timeout=5)
            if resp.status_code == 200:
                persons = resp.json()
                st.metric("등록된 인원", f"{len(persons)}명")
            else:
                st.info("인원 정보를 가져올 수 없습니다")
        except requests.ConnectionError:
            st.info("서버에 연결할 수 없습니다")
        except Exception:
            st.info("인원 정보를 가져올 수 없습니다")

        st.divider()

        # ── 오버레이 표시 옵션 ──
        st.subheader("🏷️ 오버레이 표시")
        st.session_state["overlay_show_name"] = st.checkbox(
            "이름 표시", value=st.session_state.get("overlay_show_name", True)
        )
        st.session_state["overlay_show_phone"] = st.checkbox(
            "전화번호 표시", value=st.session_state.get("overlay_show_phone", False)
        )
        st.session_state["overlay_show_address"] = st.checkbox(
            "주소 표시", value=st.session_state.get("overlay_show_address", False)
        )
