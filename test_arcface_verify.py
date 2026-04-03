"""ArcFace 모델 동작 검증 테스트."""

import cv2
import numpy as np
from deepface import DeepFace

MODEL = "ArcFace"
DETECTOR = "retinaface"


def cosine_distance(a, b):
    a, b = a.flatten(), b.flatten()
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


face_img = cv2.imread("face_db/29/1774340612674.jpg")

# ── 1. skip 모드 기본 테스트 ──
print("=== 1. ArcFace skip: face vs 단색 ===")
black = np.zeros((112, 112, 3), dtype=np.uint8)
r1 = DeepFace.represent(
    img_path=face_img,
    model_name=MODEL,
    detector_backend="skip",
    enforce_detection=False,
)
r2 = DeepFace.represent(
    img_path=black, model_name=MODEL, detector_backend="skip", enforce_detection=False
)
e_face = np.array(r1[0]["embedding"], dtype=np.float32)
e_black = np.array(r2[0]["embedding"], dtype=np.float32)
d = cosine_distance(e_face, e_black)
print(
    f"  face vs black 거리: {d:.6f} 유사도: {(1 - d) * 100:.1f}% — {'PASS' if d > 0.3 else 'FAIL'}"
)
print(f"  임베딩 dims: {len(e_face)}")

# ── 2. retinaface 모드 테스트 ──
print("\n=== 2. ArcFace retinaface ===")
r3 = DeepFace.represent(
    img_path=face_img,
    model_name=MODEL,
    detector_backend=DETECTOR,
    enforce_detection=False,
)
e_detect = np.array(r3[0]["embedding"], dtype=np.float32)
d2 = cosine_distance(e_face, e_detect)
print(f"  skip vs retinaface 거리: {d2:.6f}")
print(f"  facial_area: {r3[0].get('facial_area', 'N/A')}")

# ── 3. 같은 얼굴 좌우반전 ──
print("\n=== 3. 같은 얼굴 원본 vs 좌우반전 ===")
flipped = cv2.flip(face_img, 1)
r4 = DeepFace.represent(
    img_path=flipped, model_name=MODEL, detector_backend="skip", enforce_detection=False
)
e_flip = np.array(r4[0]["embedding"], dtype=np.float32)
d3 = cosine_distance(e_face, e_flip)
print(f"  원본 vs 좌우반전 거리: {d3:.6f} 유사도: {(1 - d3) * 100:.1f}%")

# ── 4. 등록 시뮬레이션: retinaface detect → 임베딩 추출 ──
print("\n=== 4. 등록 시뮬레이션 (retinaface detect) ===")
r5 = DeepFace.represent(
    img_path=face_img,
    model_name=MODEL,
    detector_backend=DETECTOR,
    enforce_detection=True,
)
e_reg = np.array(r5[0]["embedding"], dtype=np.float32)
print(f"  등록 임베딩 norm: {np.linalg.norm(e_reg):.4f}")
d4 = cosine_distance(e_reg, e_detect)
print(f"  등록 vs 검출 거리: {d4:.6f} — {'PASS (같음)' if d4 < 0.01 else 'CHECK'}")

# ── 5. 결론 ──
print("\n=== 결론 ===")
print(
    f"  ArcFace는 다른 입력에 대해 다른 임베딩을 생성합니까? {'YES' if d > 0.3 else 'NO'}"
)
print(
    f"  RECOGNITION_THRESHOLD=0.40 에서 face vs black 구분? {'YES' if d > 0.40 else 'NO'}"
)
print(
    f"  DUPLICATE_THRESHOLD=0.25 에서 face vs flipped 같은사람? {'YES' if d3 < 0.25 else 'NO'}"
)
