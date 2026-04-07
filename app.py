"""DeepFace Live — 서비스 테스트 Streamlit 앱.

실행: uv run streamlit run app.py
"""

import sys

sys.path.insert(0, ".")

import io

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="DeepFace Live 테스트", layout="wide")

# ──────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────
if "repo" not in st.session_state:
    st.session_state.repo = None
if "face_svc" not in st.session_state:
    st.session_state.face_svc = None
if "animal_svc" not in st.session_state:
    st.session_state.animal_svc = None
if "plant_svc" not in st.session_state:
    st.session_state.plant_svc = None


# ──────────────────────────────────────────────
# 서비스 초기화 (캐시)
# ──────────────────────────────────────────────
@st.cache_resource
def load_services():
    from server.repositories import get_repository
    from server.services.face.face_service import FaceService
    from server.services.animal.animal_service import AnimalService
    from server.services.plant.plant_service import PlantService

    repo = get_repository()

    face_svc = FaceService()
    with st.spinner("DeepFace 모델 로딩 중…"):
        face_svc.warmup()
        face_svc.load_embedding_cache(repo)

    animal_svc = AnimalService()
    with st.spinner("동물 YOLO 모델 로딩 중…"):
        animal_svc.warmup()

    plant_svc = PlantService()
    with st.spinner("식물 YOLO 모델 로딩 중…"):
        plant_svc.warmup()

    return repo, face_svc, animal_svc, plant_svc


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def bytes_to_bgr(uploaded) -> np.ndarray:
    arr = np.frombuffer(uploaded.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_bbox(img_rgb: np.ndarray, x, y, w, h, label: str, color=(0, 200, 0)) -> np.ndarray:
    out = img_rgb.copy()
    cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    cv2.putText(out, label, (int(x), int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_xyxy(img_rgb: np.ndarray, x1, y1, x2, y2, label: str, color=(255, 140, 0)) -> np.ndarray:
    out = img_rgb.copy()
    cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(out, label, (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.title("🤖 DeepFace Live — 서비스 테스트")
st.page_link("pages/animal_crossing_webcam.py", label="🐾 동물 라인 크로싱 (실시간 웹캠)", icon="📹")
st.divider()

with st.sidebar:
    st.header("서비스 초기화")
    if st.button("🔌 서비스 연결", use_container_width=True):
        with st.spinner("초기화 중…"):
            try:
                repo, face_svc, animal_svc, plant_svc = load_services()
                st.session_state.repo = repo
                st.session_state.face_svc = face_svc
                st.session_state.animal_svc = animal_svc
                st.session_state.plant_svc = plant_svc
                st.success("연결 완료")
            except Exception as e:
                st.error(f"초기화 실패: {e}")

    if st.session_state.face_svc:
        n = len(st.session_state.face_svc._embedding_cache)
        st.metric("임베딩 캐시", f"{n}명")
        st.metric("FAISS 인덱스", "✅ 준비" if st.session_state.face_svc._faiss_index else "❌ 없음")
    if st.session_state.animal_svc:
        st.metric("동물 모델", "✅ 로드됨" if st.session_state.animal_svc.is_ready() else "❌ 미로드")
    if st.session_state.plant_svc:
        st.metric("식물 모델", "✅ 로드됨" if st.session_state.plant_svc.is_ready() else "❌ 미로드")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["👤 얼굴 등록", "🔍 얼굴 인식", "🐾 동물 탐지", "🌿 식물 탐지", "📋 인물 목록", "🔴 동물 동영상 테스트"])


# ══════════════════════════════════════════════
# TAB 1 — 얼굴 등록
# ══════════════════════════════════════════════
with tab1:
    st.header("얼굴 등록")
    if not st.session_state.face_svc:
        st.warning("사이드바에서 서비스를 먼저 연결하세요.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            upload = st.file_uploader("등록할 얼굴 이미지", type=["jpg", "jpeg", "png"], key="reg_img")
        with col2:
            person_id_input = st.number_input("인물 ID (0 = 자동 생성)", min_value=0, value=0, step=1)
            condition = st.selectbox("촬영 조건", ["", "정면", "측면", "원거리", "조명어두움"])
            display_name = st.text_input("표시 이름 (선택)")

        if upload and st.button("📥 등록", use_container_width=True):
            img_bytes = upload.read()
            repo = st.session_state.repo
            face_svc = st.session_state.face_svc

            # 이미지 미리보기
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            preview = bgr_to_rgb(cv2.imdecode(arr, cv2.IMREAD_COLOR))
            st.image(preview, caption="입력 이미지", width=300)

            pid = int(person_id_input) if person_id_input > 0 else None

            # 자동 person 생성
            auto_created = False
            if pid is None:
                seq = repo.seq.next_person_number()
                p_name = f"person{seq}"
                person = repo.person.create(
                    name=p_name,
                    display_name=display_name or None,
                )
                pid = person.id
                auto_created = True
                st.info(f"새 인물 생성: **{p_name}** (ID={pid})")

            try:
                result = face_svc.register_face(
                    image_bytes=img_bytes,
                    person_id=pid,
                    repo=repo,
                    capture_condition=condition or None,
                )
                st.success(f"✅ 등록 완료 — face_image_id={result['face_image_id']}, person_id={pid}")
            except ValueError as e:
                msg = str(e)
                st.error(f"❌ 등록 실패: {msg}")
                if auto_created:
                    repo.person.delete(pid)
                    st.warning("자동 생성된 인물 롤백됨")


# ══════════════════════════════════════════════
# TAB 2 — 얼굴 인식
# ══════════════════════════════════════════════
with tab2:
    st.header("얼굴 인식")
    if not st.session_state.face_svc:
        st.warning("사이드바에서 서비스를 먼저 연결하세요.")
    else:
        upload = st.file_uploader("인식할 이미지", type=["jpg", "jpeg", "png"], key="rec_img")

        if upload and st.button("🔍 인식 실행", use_container_width=True):
            img_bytes = upload.read()
            face_svc = st.session_state.face_svc
            repo = st.session_state.repo

            results = face_svc.search_face(img_bytes, repo)

            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_rgb = bgr_to_rgb(cv2.imdecode(arr, cv2.IMREAD_COLOR))

            if not results:
                st.warning("얼굴을 감지하지 못했습니다.")
                st.image(img_rgb, width=400)
            else:
                for r in results:
                    x, y, w, h = r["bbox"]
                    name = r.get("display_name") or r.get("person_name") or "Unknown"
                    conf = r.get("confidence", 0.0)
                    label = f"{name} ({conf:.2f})"
                    color = (0, 200, 0) if r["person_id"] else (200, 0, 0)
                    img_rgb = draw_bbox(img_rgb, x, y, w, h, label, color)

                st.image(img_rgb, caption=f"감지 {len(results)}명", use_container_width=True)
                st.dataframe(
                    [
                        {
                            "person_id": r["person_id"],
                            "이름": r.get("display_name") or r.get("person_name"),
                            "confidence": round(r["confidence"], 4),
                        }
                        for r in results
                    ]
                )

        st.divider()
        if st.button("🔄 임베딩 캐시 재로드"):
            st.session_state.face_svc.reload_cache(st.session_state.repo)
            st.success(f"캐시 재로드 완료 ({len(st.session_state.face_svc._embedding_cache)}명)")


# ══════════════════════════════════════════════
# TAB 3 — 동물 탐지
# ══════════════════════════════════════════════
with tab3:
    st.header("동물 탐지")
    col1, col2 = st.columns(2)
    with col1:
        upload = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], key="animal_img")
    with col2:
        conf_thr = st.slider("Confidence 임계값", 0.1, 1.0, 0.4, 0.05)

    if upload and st.button("🐾 탐지 실행", use_container_width=True):
        import requests as _requests
        upload.seek(0)
        img_bytes = upload.read()
        img_bgr = bytes_to_bgr(io.BytesIO(img_bytes))
        img_rgb = bgr_to_rgb(img_bgr)

        # FastAPI를 통해 탐지 + DB 저장
        try:
            resp = _requests.post(
                "http://localhost:8000/api/animal/detect",
                files={"file": (upload.name, img_bytes, "image/jpeg")},
                params={"conf": conf_thr},
                timeout=30,
            )
            api_result = resp.json() if resp.ok else {}
        except Exception as e:
            st.warning(f"FastAPI 호출 실패: {e}")
            api_result = {}

        detections = api_result.get("detections", [])

        if not detections:
            st.info("탐지된 동물 없음")
            st.image(img_rgb, use_container_width=True)
        else:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['class_name']} {det['confidence']:.2f}"
                img_rgb = draw_xyxy(img_rgb, x1, y1, x2, y2, label)

            st.image(img_rgb, caption=f"탐지 {len(detections)}마리", use_container_width=True)
            st.dataframe(
                [
                    {"클래스": d["class_name"], "confidence": round(d["confidence"], 4),
                     "bbox": [round(v, 1) for v in d["bbox"]]}
                    for d in detections
                ]
            )


# ══════════════════════════════════════════════
# TAB 4 — 식물 탐지
# ══════════════════════════════════════════════
with tab4:
    st.header("식물 탐지")
    col1, col2 = st.columns(2)
    with col1:
        upload = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], key="plant_img")
    with col2:
        conf_thr = st.slider("Confidence 임계값", 0.05, 1.0, 0.15, 0.05, key="plant_conf")

    if upload and st.button("🌿 탐지 실행", use_container_width=True):
        import requests as _requests
        upload.seek(0)
        img_bytes = upload.read()
        img_bgr = bytes_to_bgr(io.BytesIO(img_bytes))
        img_rgb = bgr_to_rgb(img_bgr)

        # FastAPI를 통해 탐지 + DB 저장
        try:
            resp = _requests.post(
                "http://localhost:8000/api/plant/detect",
                files={"file": (upload.name, img_bytes, "image/jpeg")},
                params={"conf": conf_thr},
                timeout=30,
            )
            api_result = resp.json() if resp.ok else {}
        except Exception as e:
            st.warning(f"FastAPI 호출 실패: {e}")
            api_result = {}

        detections = api_result.get("detections", [])

        if not detections:
            st.info("탐지된 식물 없음")
            st.image(img_rgb, use_container_width=True)
        else:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['class_name']} {det['confidence']:.2f}"
                img_rgb = draw_xyxy(img_rgb, x1, y1, x2, y2, label, color=(34, 139, 34))

            st.image(img_rgb, caption=f"탐지 {len(detections)}개", use_container_width=True)
            st.dataframe(
                [
                    {"클래스": d["class_name"], "confidence": round(d["confidence"], 4),
                     "bbox": [round(v, 1) for v in d["bbox"]]}
                    for d in detections
                ]
            )


# ══════════════════════════════════════════════
# TAB 5 — 인물 목록
# ══════════════════════════════════════════════
with tab5:
    st.header("등록된 인물 목록")
    if not st.session_state.repo:
        st.warning("사이드바에서 서비스를 먼저 연결하세요.")
    else:
        repo = st.session_state.repo

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ 인물 삭제", use_container_width=True):
                del_id = st.session_state.get("selected_del_id")
                if del_id:
                    face_svc = st.session_state.face_svc
                    if face_svc:
                        face_svc._invalidate_cache_for_person(del_id)
                    repo.face_image.delete_by_person(del_id)
                    repo.person.delete(del_id)
                    st.success(f"ID={del_id} 삭제 완료")
                    st.rerun()

        persons = repo.person.list_all()
        if not persons:
            st.info("등록된 인물이 없습니다.")
        else:
            rows = []
            for p in persons:
                faces = repo.face_image.get_by_person(p.id)
                rows.append({
                    "ID": p.id,
                    "이름": p.name,
                    "표시 이름": p.display_name or "",
                    "얼굴 수": len(faces),
                    "전화": p.phone or "",
                    "등록일": str(p.created_at)[:19] if p.created_at else "",
                })

            st.dataframe(rows, use_container_width=True)

            del_id = st.number_input("삭제할 인물 ID", min_value=1, step=1, key="del_id_input")
            st.session_state["selected_del_id"] = int(del_id)

            # 선택한 인물 얼굴 이미지 미리보기
            sel_id = st.number_input("얼굴 이미지 조회할 인물 ID", min_value=1, step=1)
            if st.button("👁️ 얼굴 이미지 보기"):
                faces = repo.face_image.get_by_person(int(sel_id))
                if not faces:
                    st.info("등록된 얼굴 이미지가 없습니다.")
                else:
                    cols = st.columns(min(len(faces), 5))
                    for i, face in enumerate(faces[:5]):
                        try:
                            img = cv2.imread(face.image_path)
                            if img is not None:
                                cols[i].image(bgr_to_rgb(img), caption=face.capture_condition or f"face_{face.id}", width=120)
                        except Exception:
                            cols[i].warning(f"이미지 없음: {face.image_path}")


# ══════════════════════════════════════════════
# TAB 6 — 라인 크로싱 테스트
# ══════════════════════════════════════════════
with tab6:
    st.header("� 동물 탐지 테스트")
    st.caption("동영상 또는 이미지를 업로드하면 모든 프레임에서 동물을 추론하고 결과 영상을 생성합니다.")

    if not st.session_state.animal_svc:
        st.warning("사이드바에서 서비스를 먼저 연결하세요.")
    else:
        animal_svc = st.session_state.animal_svc

        # ── 입력 방식 선택 ──────────────────────────
        input_mode = st.radio(
            "입력 방식",
            ["🎬 동영상 파일", "🖼️ 이미지 시퀀스"],
            horizontal=True,
            key="crossing_mode",
        )

        video_upload = None
        imgs_upload = []

        if input_mode == "🎬 동영상 파일":
            video_upload = st.file_uploader(
                "동영상 업로드 (.mp4 / .avi / .mov)",
                type=["mp4", "avi", "mov"],
                key="crossing_video",
            )
        else:
            imgs_upload = st.file_uploader(
                "이미지 업로드 (파일명 순서대로 처리됨)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="crossing_imgs",
            )

        _opt1, _opt2 = st.columns(2)
        conf_crossing = _opt1.slider("Confidence", 0.1, 1.0, 0.4, 0.05, key="crossing_conf")
        frame_step = int(_opt2.number_input("N프레임마다 처리 (동영상만)", min_value=1, value=5, step=1, key="frame_step"))

        run_crossing = st.button("▶️ 테스트 실행", use_container_width=True, key="run_crossing")

        if run_crossing:
            # 프레임 수집
            frames_to_process: list[tuple[int, np.ndarray]] = []

            if input_mode == "🎬 동영상 파일":
                if not video_upload:
                    st.warning("동영상 파일을 먼저 업로드하세요.")
                else:
                    import os, tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as _tmp:
                        _tmp.write(video_upload.read())
                        _tmp_path = _tmp.name
                    cap = cv2.VideoCapture(_tmp_path)
                    _fi = 0
                    while True:
                        ret, _frame = cap.read()
                        if not ret:
                            break
                        if _fi % frame_step == 0:
                            frames_to_process.append((_fi, _frame))
                        _fi += 1
                    cap.release()
                    os.unlink(_tmp_path)
            else:
                if not imgs_upload:
                    st.warning("이미지를 먼저 업로드하세요.")
                else:
                    _sorted_imgs = sorted(imgs_upload, key=lambda f: f.name)
                    for _i, _img_file in enumerate(_sorted_imgs):
                        _arr = np.frombuffer(_img_file.read(), dtype=np.uint8)
                        _frame = cv2.imdecode(_arr, cv2.IMREAD_COLOR)
                        if _frame is not None:
                            frames_to_process.append((_i, _frame))

            if frames_to_process:
                import tempfile, os as _os

                total_det = 0
                det_log: list[dict] = []
                rendered_bgr: list[np.ndarray] = []

                prog_bar = st.progress(0, text="프레임 추론 중…")
                for _idx, (_fi, _frame) in enumerate(frames_to_process):
                    _det_result = animal_svc.detect(_frame, conf_threshold=conf_crossing)

                    _vis_bgr = _frame.copy()

                    # 탐지 bbox — 녹색
                    for _d in _det_result.detections:
                        _bx1, _by1, _bx2, _by2 = (int(v) for v in _d.bbox)
                        cv2.rectangle(_vis_bgr, (_bx1, _by1), (_bx2, _by2), (0, 200, 0), 2)
                        cv2.putText(_vis_bgr, f"{_d.class_name} {_d.confidence:.2f}",
                                    (_bx1, _by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                        total_det += 1
                        det_log.append({
                            "프레임": _fi,
                            "클래스": _d.class_name,
                            "confidence": round(_d.confidence, 4),
                            "bbox": [round(v, 2) for v in _d.bbox],
                        })

                    _det_n = len(_det_result)
                    cv2.putText(_vis_bgr, f"F:{_fi}  det:{_det_n}", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    rendered_bgr.append(_vis_bgr)
                    prog_bar.progress((_idx + 1) / len(frames_to_process),
                                      text=f"추론 중… {_idx + 1}/{len(frames_to_process)}")

                prog_bar.empty()

                # ── DB 저장 (FastAPI 배치 엔드포인트) ──────────────────────────
                if det_log:
                    import requests as _req
                    _payload = [
                        {"class_name": d["클래스"], "confidence": d["confidence"], "bbox": d.get("bbox", [])}
                        for d in det_log
                    ]
                    try:
                        _r = _req.post(
                            "http://localhost:8000/api/animal/logs/batch",
                            json=_payload,
                            params={"source": "video"},
                            timeout=30,
                        )
                        if _r.ok:
                            st.toast(f"DB 저장 완료: {_r.json().get('saved', 0)}건", icon="✅")
                        else:
                            st.warning(f"DB 저장 실패: {_r.text}")
                    except Exception as _e:
                        st.warning(f"DB 저장 오류: {_e}")

                # ── 결과 표시 ──────────────────────────
                st.divider()
                st.subheader("📊 결과")
                _m1, _m2 = st.columns(2)
                _m1.metric("총 탐지 수", total_det)
                _m2.metric("처리 프레임 수", len(frames_to_process))

                # 영상 생성
                _h, _w = rendered_bgr[0].shape[:2]
                _fps_out = max(1, min(30, len(rendered_bgr) // max(1, len(rendered_bgr) // 10)))
                _tmp_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                _tmp_v.close()
                _fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                _writer = cv2.VideoWriter(_tmp_v.name, _fourcc, _fps_out, (_w, _h))
                for _f in rendered_bgr:
                    _writer.write(_f)
                _writer.release()

                with open(_tmp_v.name, "rb") as _vf:
                    _video_bytes = _vf.read()
                _os.unlink(_tmp_v.name)

                st.subheader("🎬 추론 결과 영상")
                st.caption(f"{len(rendered_bgr)}프레임 · {_fps_out} fps · {_w}×{_h}")
                st.video(_video_bytes)
                st.download_button(
                    "⬇️ 결과 영상 다운로드 (MP4)",
                    data=_video_bytes,
                    file_name="animal_detection_result.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

                if det_log:
                    st.subheader("📋 탐지 로그")
                    st.dataframe(det_log, use_container_width=True)
                else:
                    st.info("탐지된 동물이 없습니다.")

