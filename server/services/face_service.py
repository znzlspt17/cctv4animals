import logging
import os
import time

import cv2
import numpy as np
from deepface import DeepFace

from server.config import settings

logger = logging.getLogger(__name__)


class FaceService:
    """DeepFace 기반 얼굴 등록/인식 서비스 — 싱글톤으로 사용."""

    def __init__(self):
        self._embedding_cache: list[dict] = []
        # 각 항목: {person_id, person_name, display_name,
        #          face_image_id, embedding(np.ndarray)}
        self._cache_loaded: bool = False
        self._alert_cache: dict[int, list[dict]] = {}

    # ──────────────────────────────────────────────
    # Warmup
    # ──────────────────────────────────────────────
    def warmup(self):
        """모델 프리로드 — lifespan에서 호출."""
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                img_path=dummy,
                model_name=settings.DEEPFACE_MODEL,
                detector_backend="skip",
                enforce_detection=False,
            )
            logger.info("DeepFace model warmed up successfully")
        except Exception:
            logger.warning(
                "Warmup produced expected error, model is loaded",
            )

    # ──────────────────────────────────────────────
    # Embedding Cache
    # ──────────────────────────────────────────────
    def load_embedding_cache(self, repo):
        """서버 시작 시 1회 호출. DB에서 전체 임베딩 캐시 로드."""
        raw = repo.face_image.get_all_embeddings()
        # raw: [{"person_id": int, "embedding": bytes}, ...]
        # person_name 조회를 위해 person 목록을 한 번에 가져옴
        persons = repo.person.list_all()
        person_map = {
            p.id: {"name": p.name, "display_name": getattr(p, "display_name", None)}
            for p in persons
        }

        self._embedding_cache = []
        for r in raw:
            if r["embedding"] is None:
                continue
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            info = person_map.get(
                r["person_id"],
                {"name": "Unknown", "display_name": None},
            )
            self._embedding_cache.append(
                {
                    "person_id": r["person_id"],
                    "person_name": info["name"],
                    "display_name": info["display_name"],
                    "face_image_id": r.get("face_image_id"),
                    "embedding": emb,
                }
            )
        self._cache_loaded = True
        logger.info("Embedding cache loaded: %d entries", len(self._embedding_cache))

    def reload_cache(self, repo):
        """캐시 무효화 후 재로드."""
        self._embedding_cache = []
        self._cache_loaded = False
        self.load_embedding_cache(repo)
        self.load_alert_cache(repo)

    def _add_to_cache(
        self,
        person_id: int,
        person_name: str,
        face_image_id: int,
        embedding: np.ndarray,
        display_name: str | None = None,
    ):
        """등록 성공 후 호출. 전체 리로드 대신 append."""
        self._embedding_cache.append(
            {
                "person_id": person_id,
                "person_name": person_name,
                "display_name": display_name,
                "face_image_id": face_image_id,
                "embedding": embedding,
            }
        )

    def _invalidate_cache_for_person(self, person_id: int):
        """인물 삭제 시 호출. 해당 person의 캐시 항목 제거."""
        self._embedding_cache = [
            e for e in self._embedding_cache if e["person_id"] != person_id
        ]
        self._alert_cache.pop(person_id, None)

    # ──────────────────────────────────────────────
    # Alert Cache
    # ──────────────────────────────────────────────
    def load_alert_cache(self, repo):
        """서버 시작 시 1회 호출. DB에서 전체 알림 규칙 캐시 로드."""
        self._alert_cache = {}
        person_ids = {e["person_id"] for e in self._embedding_cache}
        for pid in person_ids:
            try:
                rules = repo.alert_rule.get_active_rules(pid)
                if rules:
                    self._alert_cache[pid] = [
                        {"alert_type": r.alert_type, "message": r.message}
                        for r in rules
                    ]
            except Exception:
                pass
        logger.info("Alert cache loaded: %d persons with rules", len(self._alert_cache))

    def _get_cached_alerts(self, person_id: int) -> list[dict]:
        """캐시에서 알림 규칙 조회."""
        return self._alert_cache.get(person_id, [])

    # ──────────────────────────────────────────────
    # Cosine Distance (numpy 벡터화)
    # ──────────────────────────────────────────────
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 1.0
        return 1.0 - (dot / norm)

    # ──────────────────────────────────────────────
    # B-2b.1  validate_registration_image
    # ──────────────────────────────────────────────
    def validate_registration_image(self, image_bytes: bytes) -> tuple:
        """
        등록 품질 검증 (R1~R5).

        Returns:
            성공: (True, "", "", {"face": face_data, "img": img_array})
            실패: (False, error_code, error_message, {})
        """
        # numpy 이미지 디코딩
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return (False, "NO_FACE", "이미지를 디코딩할 수 없습니다", {})

        # DeepFace.extract_faces — R1: 얼굴 감지
        try:
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend=settings.DEEPFACE_DETECTOR,
                enforce_detection=True,
                align=True,
            )
        except ValueError:
            # enforce_detection=True 이므로 얼굴 미감지 시 ValueError
            return (False, "NO_FACE", "얼굴이 감지되지 않았습니다", {})

        if not faces:
            return (False, "NO_FACE", "얼굴이 감지되지 않았습니다", {})

        # R5: 단일 얼굴
        if len(faces) > 1:
            return (False, "MULTIPLE_FACES", "한 명만 촬영하세요", {})

        face = faces[0]
        confidence = face.get("confidence", 0.0)
        facial_area = face.get("facial_area", {})
        x = facial_area.get("x", 0)
        y = facial_area.get("y", 0)
        w = facial_area.get("w", 0)
        h = facial_area.get("h", 0)

        # R2: 감지 신뢰도
        if confidence < settings.FACE_MIN_CONFIDENCE:
            return (False, "LOW_CONFIDENCE", "감지 신뢰도 부족 (재촬영 필요)", {})

        # R3: 최소 얼굴 크기
        if w < settings.FACE_MIN_SIZE or h < settings.FACE_MIN_SIZE:
            return (
                False,
                "FACE_TOO_SMALL",
                "얼굴이 너무 작습니다 (가까이 촬영하세요)",
                {},
            )

        # R4: 이미지 선명도 — Laplacian variance
        # 얼굴 영역만 crop하여 선명도 측정
        face_y1 = max(0, y)
        face_y2 = min(img.shape[0], y + h)
        face_x1 = max(0, x)
        face_x2 = min(img.shape[1], x + w)
        face_crop = img[face_y1:face_y2, face_x1:face_x2]
        if face_crop.size == 0:
            return (False, "BLURRY_IMAGE", "이미지가 흐릿합니다 (재촬영 필요)", {})

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < settings.FACE_BLUR_THRESHOLD:
            return (False, "BLURRY_IMAGE", "이미지가 흐릿합니다 (재촬영 필요)", {})

        return (
            True,
            "",
            "",
            {
                "face": face,
                "img": img,
                "bbox": [x, y, w, h],
                "confidence": confidence,
            },
        )

    # ──────────────────────────────────────────────
    # B-2b.2  validate_detection_frame
    # ──────────────────────────────────────────────
    def validate_detection_frame(self, faces: list) -> list:
        """
        검출 품질 필터 (D2, D3).
        통과한 얼굴만 반환. 미통과 얼굴은 조용히 스킵.
        """
        valid = []
        for face in faces:
            confidence = face.get("confidence", face.get("face_confidence", 0.0))
            area = face.get("facial_area", {})
            w = area.get("w", 0)
            h = area.get("h", 0)

            # D2: 실시간 신뢰도
            if confidence < settings.FACE_MIN_CONFIDENCE_REALTIME:
                continue
            # D3: 실시간 최소 크기
            min_rt = settings.FACE_MIN_SIZE_REALTIME
            if w < min_rt or h < min_rt:
                continue

            valid.append(face)
        return valid

    # ──────────────────────────────────────────────
    # B-2b.3  register_face
    # ──────────────────────────────────────────────
    def register_face(
        self,
        image_bytes: bytes,
        person_id: int,
        repo,
        capture_condition: str | None = None,
        exclude_person_id: int | None = None,
    ) -> dict:
        """
        얼굴 등록: validate → extract embedding → check duplicate → save.

        Returns:
            {"face_image_id": int, "person_id": int, "message": str}

        Raises:
            ValueError: 품질 검증 실패 또는 중복 감지 시
        """
        # 1. 품질 검증
        ok, error_code, error_msg, data = self.validate_registration_image(image_bytes)
        if not ok:
            raise ValueError(f"{error_code}:{error_msg}")

        img = data["img"]
        bbox = data["bbox"]  # [x, y, w, h]

        # 2. R7: 전체 이미지에서 임베딩 추출 (detector로 정렬 보장)
        x, y, w, h = bbox
        y1, y2 = max(0, y), min(img.shape[0], y + h)
        x1, x2 = max(0, x), min(img.shape[1], x + w)
        face_crop = img[y1:y2, x1:x2]  # 이미지 저장용

        embedding_np = self._extract_embedding(img, for_registration=True)
        if embedding_np is None:
            raise ValueError("EMBEDDING_FAIL:임베딩 추출에 실패했습니다")

        # 3. R6: 중복 체크
        is_dup, matched_person_id, min_distance = self.check_duplicate(
            embedding_np,
            repo,
            exclude_person_id=exclude_person_id,
        )
        if is_dup:
            matched_person = repo.person.get(matched_person_id)
            matched_name = matched_person.name if matched_person else "Unknown"
            similarity_pct = round((1 - min_distance) * 100, 1)
            raise ValueError(
                f"DUPLICATE:이미 등록된 얼굴입니다 "
                f"(매칭: {matched_name}, 유사도: {similarity_pct}%)"
            )

        # 4. 얼굴 crop 이미지 저장
        person_dir = os.path.join(settings.FACE_DB_PATH, str(person_id))
        os.makedirs(person_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}.jpg"
        image_path = os.path.join(person_dir, filename)
        cv2.imwrite(image_path, face_crop)

        # 5. DB 저장
        embedding_bytes = embedding_np.astype(np.float32).tobytes()
        face_record = repo.face_image.create(
            person_id=person_id,
            image_path=image_path,
            embedding=embedding_bytes,
            capture_condition=capture_condition,
        )

        # 6. 캐시 갱신
        person = repo.person.get(person_id)
        person_name = person.name if person else f"Person_{person_id}"
        display_name = getattr(person, "display_name", None) if person else None
        self._add_to_cache(
            person_id=person_id,
            person_name=person_name,
            face_image_id=face_record.id,
            embedding=embedding_np,
            display_name=display_name,
        )

        logger.info(
            "Face registered: person_id=%d, face_id=%d",
            person_id,
            face_record.id,
        )
        return {
            "face_image_id": face_record.id,
            "person_id": person_id,
            "message": "등록 완료",
        }

    # ──────────────────────────────────────────────
    # B-2b.4  search_face
    # ──────────────────────────────────────────────
    def search_face(self, image_bytes: bytes, repo) -> list:
        """
        실시간 얼굴 인식. 프레임에서 얼굴을 검출하고 캐시된 임베딩과 비교.

        Returns:
            list[dict] — RecognitionResult 호환 딕셔너리 목록.
            얼굴 미감지 시 빈 리스트 반환 (에러 없음).
        """
        # numpy 이미지 디코딩
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        # D1: 얼굴 검출 + 임베딩 추출 (represent 한 번에 수행하여 정렬 보장)
        try:
            representations = DeepFace.represent(
                img_path=img,
                model_name=settings.DEEPFACE_MODEL,
                detector_backend=settings.DEEPFACE_DETECTOR_REALTIME,
                enforce_detection=False,
                align=True,
            )
        except Exception as e:
            logger.debug("Face detection/embedding failed: %s", e)
            return []

        if not representations:
            return []

        # D2, D3: 품질 필터
        valid_reps = self.validate_detection_frame(representations)
        if not valid_reps:
            return []

        results = []
        for rep in valid_reps:
            area = rep.get("facial_area", {})
            bbox = [
                area.get("x", 0),
                area.get("y", 0),
                area.get("w", 0),
                area.get("h", 0),
            ]

            embedding_np = np.array(rep["embedding"], dtype=np.float32)

            # 캐시 매칭
            if not self._embedding_cache:
                results.append(
                    {
                        "person_id": None,
                        "person_name": "Unknown",
                        "display_name": None,
                        "confidence": 0.0,
                        "bbox": bbox,
                        "alerts": [],
                    }
                )
                continue

            # numpy 벡터화 cosine distance
            db_matrix = np.array([e["embedding"] for e in self._embedding_cache])
            new_vec = embedding_np.flatten()
            norms_db = np.linalg.norm(db_matrix, axis=1)
            norm_new = np.linalg.norm(new_vec)
            denominator = norms_db * norm_new + 1e-10
            cosine_sim = np.dot(db_matrix, new_vec) / denominator
            distances = 1 - cosine_sim

            min_idx = int(np.argmin(distances))
            min_distance = float(distances[min_idx])

            # D5: threshold 비교
            if min_distance <= settings.RECOGNITION_THRESHOLD:
                matched = self._embedding_cache[min_idx]
                person_id = matched["person_id"]
                person_name = matched["person_name"]
                display_name = matched.get("display_name")
                alerts = self._get_cached_alerts(person_id)

                results.append(
                    {
                        "person_id": person_id,
                        "person_name": person_name,
                        "display_name": display_name,
                        "confidence": round(1 - min_distance, 4),
                        "bbox": bbox,
                        "alerts": alerts,
                    }
                )
            else:
                # Unknown
                results.append(
                    {
                        "person_id": None,
                        "person_name": "Unknown",
                        "display_name": None,
                        "confidence": round(1 - min_distance, 4),
                        "bbox": bbox,
                        "alerts": [],
                    }
                )

        return results

    # ──────────────────────────────────────────────
    # B-2b.5  check_duplicate
    # ──────────────────────────────────────────────
    def check_duplicate(
        self,
        embedding,
        repo,
        exclude_person_id: int | None = None,
    ) -> tuple[bool, int | None, float]:
        """
        중복 감지 (R6). numpy 벡터화 cosine distance.

        Returns:
            (is_duplicate, matched_person_id, min_distance)
        """
        # 캐시에서 비교 대상 필터링
        candidates = self._embedding_cache
        if exclude_person_id is not None:
            candidates = [e for e in candidates if e["person_id"] != exclude_person_id]

        if not candidates:
            return (False, None, float("inf"))

        new_vec = np.array(embedding).flatten()
        db_matrix = np.array([c["embedding"].flatten() for c in candidates])

        # cosine distance = 1 - cosine_similarity
        norms_db = np.linalg.norm(db_matrix, axis=1)
        norm_new = np.linalg.norm(new_vec)
        denominator = norms_db * norm_new + 1e-10
        cosine_sim = np.dot(db_matrix, new_vec) / denominator
        distances = 1 - cosine_sim

        min_idx = int(np.argmin(distances))
        min_distance = float(distances[min_idx])
        matched_person_id = candidates[min_idx]["person_id"]

        is_duplicate = min_distance <= settings.DUPLICATE_THRESHOLD
        return (is_duplicate, matched_person_id, min_distance)

    # ──────────────────────────────────────────────
    # B-2b.6  register_multi_angle
    # ──────────────────────────────────────────────
    def register_multi_angle(
        self,
        images_list: list[bytes],
        repo,
        person_id: int | None = None,
    ) -> dict:
        """
        다중 각도 등록.
        - 첫 이미지만 전체 DB 대상 check_duplicate 수행.
        - 나머지 이미지는 same-session 간 중복 체크 스킵 (validate만).

        Returns:
            {"person_id": int, "registered": list, "failed": list}
        """
        if not images_list:
            raise ValueError("NO_IMAGES:등록할 이미지가 없습니다")

        registered = []
        failed = []

        for idx, image_bytes in enumerate(images_list):
            try:
                if idx == 0:
                    # 첫 이미지: 전체 DB 대상 중복 체크
                    result = self.register_face(
                        image_bytes=image_bytes,
                        person_id=person_id,
                        repo=repo,
                        capture_condition=f"multi_angle_{idx}",
                    )
                else:
                    # 나머지: 같은 person 제외하고 중복 체크 (exclude_person_id)
                    result = self.register_face(
                        image_bytes=image_bytes,
                        person_id=person_id,
                        repo=repo,
                        capture_condition=f"multi_angle_{idx}",
                        exclude_person_id=person_id,
                    )
                registered.append(result)
            except ValueError as e:
                failed.append({"index": idx, "error": str(e)})

        return {
            "person_id": person_id,
            "registered": registered,
            "failed": failed,
        }

    # ──────────────────────────────────────────────
    # delete_person_faces — 캐시 무효화 포함
    # ──────────────────────────────────────────────
    def delete_person_faces(self, person_id: int, repo):
        """인물의 얼굴 이미지/임베딩 삭제 및 캐시 무효화."""
        repo.face_image.delete_by_person(person_id)
        self._invalidate_cache_for_person(person_id)
        logger.info("Deleted faces and cache for person_id=%d", person_id)

    # ──────────────────────────────────────────────
    # 내부 헬퍼: 임베딩 추출
    # ──────────────────────────────────────────────
    def _extract_embedding(
        self,
        img: np.ndarray,
        for_registration: bool = True,
    ) -> np.ndarray | None:
        """
        DeepFace.represent()로 임베딩 추출.

        Args:
            img: BGR numpy array
            for_registration: True면 등록용 (enforce_detection=False, 이미 검증됨),
                            False면 실시간용
        """
        try:
            detector = (
                settings.DEEPFACE_DETECTOR
                if for_registration
                else settings.DEEPFACE_DETECTOR_REALTIME
            )
            results = DeepFace.represent(
                img_path=img,
                model_name=settings.DEEPFACE_MODEL,
                detector_backend=detector,
                enforce_detection=for_registration,
                align=True,
            )
            if results and len(results) > 0:
                emb = np.array(results[0]["embedding"], dtype=np.float32)
                return emb
        except Exception as e:
            logger.error("Embedding extraction failed: %s", e)
        return None

    def _extract_embedding_from_face(
        self,
        face: dict,
        img: np.ndarray,
    ) -> np.ndarray | None:
        """
        이미 추출된 face 영역에서 임베딩 추출.
        얼굴 영역을 crop 후 detector_backend="skip"으로 represent 호출.
        """
        area = face.get("facial_area", {})
        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", 0)
        h = area.get("h", 0)

        # 얼굴 영역 crop
        y1 = max(0, y)
        y2 = min(img.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(img.shape[1], x + w)
        face_crop = img[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        try:
            results = DeepFace.represent(
                img_path=face_crop,
                model_name=settings.DEEPFACE_MODEL,
                detector_backend="skip",
                enforce_detection=False,
                align=True,
            )
            if results and len(results) > 0:
                return np.array(results[0]["embedding"], dtype=np.float32)
        except Exception as e:
            logger.debug("Embedding extraction from face crop failed: %s", e)
        return None

    def _extract_embedding_from_crop(
        self,
        face_crop: np.ndarray,
    ) -> np.ndarray | None:
        """
        이미 crop된 얼굴 이미지에서 임베딩 추출.
        detector_backend="skip" — 이미 검증된 얼굴 영역.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        try:
            results = DeepFace.represent(
                img_path=face_crop,
                model_name=settings.DEEPFACE_MODEL,
                detector_backend="skip",
                enforce_detection=False,
                align=True,
            )
            if results and len(results) > 0:
                return np.array(
                    results[0]["embedding"],
                    dtype=np.float32,
                )
        except Exception as e:
            logger.error("Embedding from crop failed: %s", e)
        return None


# ── 싱글톤 인스턴스 ──
face_service = FaceService()
