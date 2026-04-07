"""두 가지 다른 얼굴 이미지의 임베딩 차이를 검증하는 테스트."""

import cv2
import numpy as np
from deepface import DeepFace

MODEL = "Buffalo_L"
DETECTOR = "retinaface"


def cosine_distance(a, b):
    a, b = a.flatten(), b.flatten()
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# ── 1. 모델 기본 동작 확인: 단색 이미지 2개 ──
print("=== 1. 단색 이미지 임베딩 비교 (skip) ===")
black = np.zeros((112, 112, 3), dtype=np.uint8)
white = np.ones((112, 112, 3), dtype=np.uint8) * 255
r_black = DeepFace.represent(
    img_path=black, model_name=MODEL, detector_backend="skip", enforce_detection=False
)
r_white = DeepFace.represent(
    img_path=white, model_name=MODEL, detector_backend="skip", enforce_detection=False
)
emb_black = np.array(r_black[0]["embedding"], dtype=np.float32)
emb_white = np.array(r_white[0]["embedding"], dtype=np.float32)
d = cosine_distance(emb_black, emb_white)
print(f"  black vs white 거리: {d:.6f} (유사도: {(1 - d) * 100:.1f}%)")

# ── 2. 랜덤 노이즈 vs 기존 이미지 (skip) ──
print("\n=== 2. 랜덤 노이즈 vs face_db 이미지 (skip) ===")
np.random.seed(42)
noise = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
r_noise = DeepFace.represent(
    img_path=noise, model_name=MODEL, detector_backend="skip", enforce_detection=False
)
emb_noise = np.array(r_noise[0]["embedding"], dtype=np.float32)

face_img = cv2.imread("face_db/29/1774340612674.jpg")
r_face = DeepFace.represent(
    img_path=face_img,
    model_name=MODEL,
    detector_backend="skip",
    enforce_detection=False,
)
emb_face = np.array(r_face[0]["embedding"], dtype=np.float32)
d2 = cosine_distance(emb_noise, emb_face)
print(f"  noise vs face 거리: {d2:.6f} (유사도: {(1 - d2) * 100:.1f}%)")

# ── 3. 같은 이미지 좌우반전 (skip) ──
print("\n=== 3. 원본 vs 좌우반전 (skip) ===")
face_flipped = cv2.flip(face_img, 1)
r_flip = DeepFace.represent(
    img_path=face_flipped,
    model_name=MODEL,
    detector_backend="skip",
    enforce_detection=False,
)
emb_flip = np.array(r_flip[0]["embedding"], dtype=np.float32)
d3 = cosine_distance(emb_face, emb_flip)
print(f"  원본 vs 좌우반전 거리: {d3:.6f} (유사도: {(1 - d3) * 100:.1f}%)")

# ── 4. retinaface detector 사용 ──
print("\n=== 4. face_db 이미지 (retinaface) ===")
try:
    r_detect = DeepFace.represent(
        img_path=face_img,
        model_name=MODEL,
        detector_backend=DETECTOR,
        enforce_detection=False,
    )
    emb_detect = np.array(r_detect[0]["embedding"], dtype=np.float32)
    d4 = cosine_distance(emb_face, emb_detect)
    print(f"  skip vs retinaface 거리: {d4:.6f}")
    print(f"  retinaface facial_area: {r_detect[0].get('facial_area', 'N/A')}")
except Exception as e:
    print(f"  retinaface 실패: {e}")

# ── 5. 핵심 테스트: 112x112로 리사이즈된 다양한 이미지에 대한 임베딩 분포 ──
print("\n=== 5. 다양한 112x112 이미지의 임베딩 분포 ===")
test_images = []
# 5a: 순수 검은색
test_images.append(("pure_black", np.zeros((112, 112, 3), dtype=np.uint8)))
# 5b: 순수 흰색
test_images.append(("pure_white", np.ones((112, 112, 3), dtype=np.uint8) * 255))
# 5c: 그레이
test_images.append(("gray", np.ones((112, 112, 3), dtype=np.uint8) * 128))
# 5d: 피부색 계열
skin = np.ones((112, 112, 3), dtype=np.uint8) * np.array(
    [140, 180, 210], dtype=np.uint8
)
test_images.append(("skin_color", skin))
# 5e: 기존 얼굴
test_images.append(("face_29", cv2.resize(face_img, (112, 112))))
# 5f: 반전
test_images.append(("face_29_flip", cv2.resize(face_flipped, (112, 112))))

embeddings = {}
for name, img in test_images:
    r = DeepFace.represent(
        img_path=img, model_name=MODEL, detector_backend="skip", enforce_detection=False
    )
    embeddings[name] = np.array(r[0]["embedding"], dtype=np.float32)

for i, (n1, e1) in enumerate(embeddings.items()):
    for n2, e2 in list(embeddings.items())[i + 1 :]:
        d = cosine_distance(e1, e2)
        print(f"  {n1} vs {n2}: 거리={d:.6f} 유사도={(1 - d) * 100:.1f}%")

print("\n=== 완료 ===")
