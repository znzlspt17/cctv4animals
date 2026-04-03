"""인물 관리 페이지."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import requests
import streamlit as st

from server.config import settings
from ui.components.sidebar import render_sidebar

st.set_page_config(page_title="인물 관리 — DeepFace Live", layout="wide")
render_sidebar()

st.title("👤 인물 관리")

_api_host = settings.FASTAPI_HOST if settings.FASTAPI_HOST != "0.0.0.0" else "localhost"
_default_api_base = f"http://{_api_host}:{settings.FASTAPI_PORT}/api"
API_BASE = st.session_state.get("api_base_url", _default_api_base)


def load_persons() -> list[dict]:
    """GET /api/persons."""
    try:
        resp = requests.get(f"{API_BASE}/persons", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        st.error("API 서버에 연결할 수 없습니다.")
    except Exception as e:
        st.error(f"오류: {e}")
    return []


def load_person_detail(person_id: int) -> dict | None:
    """GET /api/persons/{id}."""
    try:
        resp = requests.get(f"{API_BASE}/persons/{person_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            st.warning("해당 인물을 찾을 수 없습니다.")
    except requests.ConnectionError:
        st.error("API 서버에 연결할 수 없습니다.")
    except Exception as e:
        st.error(f"오류: {e}")
    return None


# ── 인물 목록 ──
persons = load_persons()

if not persons:
    st.info("등록된 인물이 없습니다.")
    st.stop()

# 목록 뷰
st.subheader(f"📋 등록된 인물 ({len(persons)}명)")

person_table_data = []
for p in persons:
    person_table_data.append(
        {
            "ID": p["id"],
            "이름": p["name"],
            "표시이름": p.get("display_name", "-") or "-",
            "전화번호": p.get("phone", "-") or "-",
            "주소": p.get("address", "-") or "-",
            "등록일": p.get("created_at", "")[:10],
        }
    )
st.dataframe(person_table_data, use_container_width=True)

st.divider()

# ── 인물 상세 / 수정 / 삭제 ──
st.subheader("🔍 인물 상세 정보")

person_options = {f"{p['name']} (ID:{p['id']})": p["id"] for p in persons}
selected_label = st.selectbox("인물 선택", list(person_options.keys()))

if selected_label:
    person_id = person_options[selected_label]
    detail = load_person_detail(person_id)

    if detail:
        col_detail, col_images = st.columns([1, 1])

        with col_detail:
            st.markdown("#### 기본 정보")
            st.text(f"ID: {detail['id']}")
            st.text(f"이름: {detail['name']}")
            st.text(f"표시이름: {detail.get('display_name') or '-'}")
            st.text(f"전화번호: {detail.get('phone') or '-'}")
            st.text(f"주소: {detail.get('address') or '-'}")
            st.text(f"등록일: {detail.get('created_at', '')[:19]}")
            if detail.get("updated_at"):
                st.text(f"수정일: {detail['updated_at'][:19]}")

        with col_images:
            st.markdown("#### 등록된 얼굴 이미지")
            face_images = detail.get("face_images", [])
            if face_images:
                img_cols = st.columns(min(len(face_images), 4))
                for i, fi in enumerate(face_images):
                    with img_cols[i % len(img_cols)]:
                        image_path = fi.get("image_path", "")
                        st.text(f"#{fi['id']} ({fi.get('capture_condition', '-')})")
                        # 이미지 파일 경로가 서버 로컬이므로 파일 URL 또는 경로만 표시
                        st.text(f"📁 {image_path}")
            else:
                st.info("등록된 이미지가 없습니다.")

        st.divider()

        # ── 정보 수정 ──
        st.markdown("#### ✏️ 정보 수정")
        with st.form(key=f"edit_person_{person_id}"):
            edit_display_name = st.text_input(
                "표시 이름", value=detail.get("display_name") or ""
            )
            edit_phone = st.text_input("전화번호", value=detail.get("phone") or "")
            edit_address = st.text_input("주소", value=detail.get("address") or "")
            submitted = st.form_submit_button("💾 저장")

            if submitted:
                update_data = {}
                if edit_display_name:
                    update_data["display_name"] = edit_display_name
                if edit_phone:
                    update_data["phone"] = edit_phone
                if edit_address:
                    update_data["address"] = edit_address

                if update_data:
                    try:
                        resp = requests.put(
                            f"{API_BASE}/persons/{person_id}",
                            json=update_data,
                            timeout=5,
                        )
                        if resp.status_code == 200:
                            st.success("✅ 저장 완료!")
                            st.rerun()
                        else:
                            st.error(f"저장 실패 (HTTP {resp.status_code})")
                    except requests.ConnectionError:
                        st.error("API 서버에 연결할 수 없습니다.")
                    except Exception as e:
                        st.error(f"오류: {e}")
                else:
                    st.info("변경사항이 없습니다.")

        st.divider()

        # ── 삭제 ──
        st.markdown("#### 🗑️ 인물 삭제")
        st.warning("⚠️ 삭제하면 해당 인물의 모든 등록 이미지와 데이터가 삭제됩니다.")

        confirm_name = st.text_input(
            f'삭제하려면 인물 이름 "{detail["name"]}"을 입력하세요',
            key=f"delete_confirm_{person_id}",
        )

        if st.button("🗑️ 삭제 실행", key=f"btn_delete_{person_id}", type="primary"):
            if confirm_name == detail["name"]:
                try:
                    resp = requests.delete(
                        f"{API_BASE}/persons/{person_id}",
                        timeout=5,
                    )
                    if resp.status_code == 204:
                        st.success("✅ 삭제 완료!")
                        st.rerun()
                    else:
                        st.error(f"삭제 실패 (HTTP {resp.status_code})")
                except requests.ConnectionError:
                    st.error("API 서버에 연결할 수 없습니다.")
                except Exception as e:
                    st.error(f"오류: {e}")
            else:
                st.error("인물 이름이 일치하지 않습니다.")
