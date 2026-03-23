"""얼굴 등록 페이지."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import requests
import streamlit as st

from ui.components.sidebar import render_sidebar

st.set_page_config(page_title="얼굴 등록 — DeepFace Live", layout="wide")
render_sidebar()

st.title("📸 얼굴 등록")

API_BASE = st.session_state.get("api_base_url", "http://localhost:8000/api")

# ── 에러 코드 → 한국어 메시지 매핑 ──
ERROR_MESSAGES = {
    "NO_FACE": "얼굴이 감지되지 않았습니다.",
    "MULTIPLE_FACES": "여러 얼굴이 감지되었습니다. 한 명만 촬영하세요.",
    "LOW_CONFIDENCE": "감지 신뢰도가 낮습니다. 정면으로 다시 촬영하세요.",
    "FACE_TOO_SMALL": "얼굴이 너무 작습니다. 카메라에 가까이 다가가세요.",
    "BLURRY_IMAGE": "이미지가 흐릿합니다. 안정적으로 다시 촬영하세요.",
    "EMBEDDING_FAIL": "얼굴 처리 중 오류가 발생했습니다. 다시 시도하세요.",
    "VALIDATION_ERROR": "입력값이 올바르지 않습니다.",
}

# ── 등록 대상 선택 ──
st.subheader("1️⃣ 등록 대상 선택")

register_mode = st.radio(
    "등록 모드", ["새 인물 등록", "기존 인물에 추가"], horizontal=True
)

selected_person_id = None
selected_person_name = ""

if register_mode == "기존 인물에 추가":
    try:
        resp = requests.get(f"{API_BASE}/persons", timeout=5)
        if resp.status_code == 200:
            persons = resp.json()
            if persons:
                person_options = {
                    f"{p['name']} (ID:{p['id']})": p["id"] for p in persons
                }
                selected = st.selectbox("인물 선택", list(person_options.keys()))
                if selected:
                    selected_person_id = person_options[selected]
                    selected_person_name = selected
            else:
                st.info("등록된 인물이 없습니다. '새 인물 등록'을 이용해주세요.")
        else:
            st.warning("인물 목록을 불러올 수 없습니다.")
    except requests.ConnectionError:
        st.error("API 서버에 연결할 수 없습니다.")
    except Exception as e:
        st.error(f"오류: {e}")

# ── 추가 정보 입력 ──
st.subheader("2️⃣ 추가 정보 (선택사항)")
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    input_display_name = st.text_input("표시 이름", placeholder="예: 홍길동")
with col_info2:
    input_phone = st.text_input("전화번호", placeholder="예: 010-1234-5678")
with col_info3:
    input_address = st.text_input("주소", placeholder="예: 서울시 강남구")

# ── 이미지 촬영/업로드 ──
st.subheader("3️⃣ 이미지 촬영 또는 업로드")

tab_camera, tab_upload, tab_multi = st.tabs(
    ["📷 웹캠 촬영", "📁 이미지 업로드", "🔄 다중 각도 등록"]
)


def _handle_register_response(resp):
    """등록 API 응답 처리."""
    if resp.status_code == 201:
        data = resp.json()
        st.success(
            f"✅ 등록 성공! 인물: {data['person_name']} (ID: {data['person_id']})"
        )
        # 추가 정보가 있으면 인물 정보 업데이트
        person_id = data["person_id"]
        update_data = {}
        if input_display_name:
            update_data["display_name"] = input_display_name
        if input_phone:
            update_data["phone"] = input_phone
        if input_address:
            update_data["address"] = input_address
        if update_data:
            try:
                requests.put(
                    f"{API_BASE}/persons/{person_id}",
                    json=update_data,
                    timeout=5,
                )
            except Exception:
                pass
    elif resp.status_code == 409:
        # 중복 얼굴
        data = resp.json()
        detail = data.get("detail", "")
        st.warning(f"⚠️ 이미 등록된 인물입니다. {detail}")
    elif resp.status_code == 400:
        data = resp.json()
        detail = data.get("detail", "")
        error_code = data.get("error_code", "")
        # error_code가 detail 안에 포함된 경우 파싱
        for code, msg in ERROR_MESSAGES.items():
            if code in error_code or code in detail:
                st.error(f"❌ {msg}")
                return
        st.error(f"❌ 등록 실패: {detail}")
    else:
        st.error(f"❌ 등록 실패 (HTTP {resp.status_code})")


def _register_single(image_bytes: bytes, filename: str):
    """단일 이미지 등록."""
    form_data = {}
    if selected_person_id is not None:
        form_data["person_id"] = str(selected_person_id)

    try:
        resp = requests.post(
            f"{API_BASE}/register",
            files={"file": (filename, image_bytes, "image/jpeg")},
            data=form_data,
            timeout=30,
        )
        _handle_register_response(resp)
    except requests.ConnectionError:
        st.error("❌ API 서버에 연결할 수 없습니다.")
    except Exception as e:
        st.error(f"❌ 오류: {e}")


# ── 탭 1: 웹캠 촬영 ──
with tab_camera:
    camera_image = st.camera_input("얼굴을 촬영하세요")
    if camera_image is not None:
        st.image(camera_image, caption="촬영된 이미지", width=300)
        if st.button("📤 촬영 이미지로 등록", key="btn_register_camera"):
            _register_single(camera_image.getvalue(), "camera_capture.jpg")

# ── 탭 2: 이미지 업로드 ──
with tab_upload:
    uploaded_file = st.file_uploader(
        "이미지 파일을 선택하세요",
        type=["jpg", "jpeg", "png"],
        key="single_upload",
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="업로드된 이미지", width=300)
        if st.button("📤 업로드 이미지로 등록", key="btn_register_upload"):
            _register_single(uploaded_file.getvalue(), uploaded_file.name)

# ── 탭 3: 다중 각도 등록 ──
with tab_multi:
    st.markdown(
        """
        **다중 각도 등록**: 여러 각도(정면, 좌측 45°, 우측 45° 등)의 이미지를
        한 번에 등록하면 인식 정확도가 높아집니다.
        """
    )
    multi_files = st.file_uploader(
        "여러 이미지 파일을 선택하세요 (최소 2장)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multi_upload",
    )

    if multi_files:
        cols = st.columns(min(len(multi_files), 4))
        for i, f in enumerate(multi_files):
            with cols[i % len(cols)]:
                st.image(f, caption=f"이미지 {i + 1}", width=150)

        if st.button("📤 다중 각도 등록 실행", key="btn_register_multi"):
            form_data = {}
            if selected_person_id is not None:
                form_data["person_id"] = str(selected_person_id)

            files_payload = [
                ("files", (f.name, f.getvalue(), "image/jpeg")) for f in multi_files
            ]

            try:
                resp = requests.post(
                    f"{API_BASE}/register/multi-angle",
                    files=files_payload,
                    data=form_data,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    registered = data.get("registered", 0)
                    failed = data.get("failed", 0)
                    st.success(
                        f"✅ 다중 각도 등록 완료! "
                        f"성공: {registered}장, 실패: {failed}장"
                    )
                else:
                    _handle_register_response(resp)
            except requests.ConnectionError:
                st.error("❌ API 서버에 연결할 수 없습니다.")
            except Exception as e:
                st.error(f"❌ 오류: {e}")
