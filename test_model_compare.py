"""다양한 모델 비교 + insightface 직접 테스트."""

import cv2
import numpy as np


def cosine_distance(a, b):
    a, b = a.flatten(), b.flatten()
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


face_img = cv2.imread("face_db/29/1774340612674.jpg")
black = np.zeros((112, 112, 3), dtype=np.uint8)

# ── 1. DeepFace + ArcFace (다른 모델) ──
print("=== 1. DeepFace + ArcFace ===")
from deepface import DeepFace

try:
    r1 = DeepFace.represent(
        img_path=face_img,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False,
    )
    r2 = DeepFace.represent(
        img_path=black,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False,
    )
    e1 = np.array(r1[0]["embedding"], dtype=np.float32)
    e2 = np.array(r2[0]["embedding"], dtype=np.float32)
    d = cosine_distance(e1, e2)
    print(f"  face vs black 거리: {d:.6f} 유사도: {(1 - d) * 100:.1f}%")
except Exception as e:
    print(f"  실패: {e}")

# ── 2. DeepFace + Facenet512 ──
print("\n=== 2. DeepFace + Facenet512 ===")
try:
    r1 = DeepFace.represent(
        img_path=face_img,
        model_name="Facenet512",
        detector_backend="skip",
        enforce_detection=False,
    )
    r2 = DeepFace.represent(
        img_path=black,
        model_name="Facenet512",
        detector_backend="skip",
        enforce_detection=False,
    )
    e1 = np.array(r1[0]["embedding"], dtype=np.float32)
    e2 = np.array(r2[0]["embedding"], dtype=np.float32)
    d = cosine_distance(e1, e2)
    print(f"  face vs black 거리: {d:.6f} 유사도: {(1 - d) * 100:.1f}%")
except Exception as e:
    print(f"  실패: {e}")

# ── 3. InsightFace 직접 사용 ──
print("\n=== 3. InsightFace 직접 사용 ===")
try:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 얼굴 이미지 (BGR -> RGB)
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    faces = app.get(face_rgb)
    print(f"  face_img: 감지된 얼굴 수={len(faces)}")
    if faces:
        emb1 = faces[0].embedding
        print(f"  임베딩 norm: {np.linalg.norm(emb1):.4f}, dims: {len(emb1)}")
        print(f"  첫 10값: {emb1[:10]}")

    # 검은 이미지로 테스트
    black_large = np.zeros((640, 640, 3), dtype=np.uint8)
    faces_black = app.get(black_large)
    print(f"  black: 감지된 얼굴 수={len(faces_black)}")

    # 두 번째 얼굴 이미지 (좌우반전)
    flipped = cv2.cvtColor(cv2.flip(face_img, 1), cv2.COLOR_BGR2RGB)
    faces_flip = app.get(flipped)
    print(f"  flipped: 감지된 얼굴 수={len(faces_flip)}")
    if faces and faces_flip:
        emb2 = faces_flip[0].embedding
        d = cosine_distance(emb1, emb2)
        print(f"  원본 vs 좌우반전 거리: {d:.6f} 유사도: {(1 - d) * 100:.1f}%")

except Exception as e:
    print(f"  실패: {e}")
    import traceback

    traceback.print_exc()

print("\n=== 완료 ===")
