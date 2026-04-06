"""진단 스크립트: 두 이미지의 임베딩 비교 + 다양한 방식 테스트."""

import glob

import numpy as np
from deepface import DeepFace

MODEL = "Buffalo_L"
DETECTOR = "retinaface"

# face_db에서 기존 등록 이미지 확인
existing = glob.glob("face_db/**/*.jpg", recursive=True)
print(f"=== face_db 기존 이미지: {existing} ===\n")

# 기존 등록 이미지 임베딩 (다양한 방식)
if existing:
    img_path = existing[0]
    print(f"--- 기존 이미지: {img_path} ---")

    # 방법1: detector=skip (broken)
    r1 = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL,
        detector_backend="skip",
        enforce_detection=False,
    )
    emb_skip = np.array(r1[0]["embedding"], dtype=np.float32)
    print(f"  skip 임베딩 norm: {np.linalg.norm(emb_skip):.4f}, dims: {len(emb_skip)}")

    # 방법2: detector=retinaface
    r2 = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL,
        detector_backend=DETECTOR,
        enforce_detection=False,
    )
    emb_detect = np.array(r2[0]["embedding"], dtype=np.float32)
    print(
        f"  retinaface 임베딩 norm: {np.linalg.norm(emb_detect):.4f}, dims: {len(emb_detect)}"
    )

    # skip vs retinaface 거리
    cos_dist = 1 - np.dot(emb_skip, emb_detect) / (
        np.linalg.norm(emb_skip) * np.linalg.norm(emb_detect)
    )
    print(
        f"  skip vs retinaface 거리: {cos_dist:.6f} (유사도: {(1 - cos_dist) * 100:.1f}%)"
    )

    # DB에 저장된 임베딩 확인
    from server.database import SessionLocal
    from server.models import FaceImage

    db = SessionLocal()
    fi = db.query(FaceImage).first()
    if fi and fi.embedding:
        db_emb = np.frombuffer(fi.embedding, dtype=np.float32)

        dist_vs_skip = 1 - np.dot(db_emb, emb_skip) / (
            np.linalg.norm(db_emb) * np.linalg.norm(emb_skip) + 1e-10
        )
        dist_vs_detect = 1 - np.dot(db_emb, emb_detect) / (
            np.linalg.norm(db_emb) * np.linalg.norm(emb_detect) + 1e-10
        )
        print(f"\n  DB임베딩 norm: {np.linalg.norm(db_emb):.4f}, dims: {len(db_emb)}")
        print(
            f"  DB vs skip 거리: {dist_vs_skip:.6f} (유사도: {(1 - dist_vs_skip) * 100:.1f}%)"
        )
        print(
            f"  DB vs retinaface 거리: {dist_vs_detect:.6f} (유사도: {(1 - dist_vs_detect) * 100:.1f}%)"
        )
    db.close()

print("\n=== 완료 ===")
