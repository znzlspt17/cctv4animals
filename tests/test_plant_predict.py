"""PlantService 실제 추론 진단 스크립트.

사용법:
    uv run python -m pytest tests/test_plant_predict.py -v -s

딸기 validation 이미지에 대해 실제 모델 출력(bbox, class, score)을 출력하고,
GT(JSON 어노테이션)의 bbox와 겹침(IoU)도 계산한다.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ──────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────
SAMPLE_DIR = Path(r"C:\Users\user\Downloads\strawberry_validation\strawberry_validation")
CONF_THRESHOLD = 0.3   # 낮게 설정해 더 많은 결과 확인


def _collect_samples(max_n: int = 10) -> list[tuple[Path, Path]]:
    """(jpg, json) 쌍을 최대 max_n개 반환."""
    pairs = []
    for jpg in sorted(SAMPLE_DIR.glob("*.jpg"))[:max_n]:
        js = jpg.with_suffix(".jpg.json")
        if js.exists():
            pairs.append((jpg, js))
    return pairs


def _iou(boxA: list, boxB: list) -> float:
    """IoU 계산. boxA/B = [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


# ──────────────────────────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def plant_svc():
    from server.services.plant.plant_service import PlantService
    svc = PlantService()
    svc.load()
    return svc


def test_model_loaded(plant_svc):
    """모델이 정상 로드되는지 확인."""
    assert plant_svc.is_ready(), "PlantService가 로드되지 않았습니다."
    print(f"\n✅ 모델 로드 완료  |  클래스: {plant_svc._class_names}")


def test_predict_samples(plant_svc):
    """실제 딸기 이미지로 predict 결과를 출력한다."""
    pairs = _collect_samples(max_n=10)
    assert pairs, f"샘플 이미지를 찾을 수 없습니다: {SAMPLE_DIR}"

    print(f"\n총 {len(pairs)}개 샘플 처리\n{'='*70}")

    hit_count = 0  # conf >= threshold & IoU >= 0.3 인 건수

    for jpg_path, json_path in pairs:
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        # GT 박스 (원본 해상도 기준)
        pts = meta["annotations"]["points"]
        gt_boxes = [[p["xtl"], p["ytl"], p["xbr"], p["ybr"]] for p in pts]
        disease_code = meta["annotations"]["disease"]

        # 이미지 로드
        frame = cv2.imread(str(jpg_path))
        assert frame is not None, f"이미지 로드 실패: {jpg_path}"
        h, w = frame.shape[:2]

        # 추론
        result = plant_svc.detect(frame, conf_threshold=CONF_THRESHOLD)

        print(f"\n📷 {jpg_path.name}")
        print(f"   해상도: {w}×{h}  |  disease code: {disease_code}")
        print(f"   GT boxes ({len(gt_boxes)}개):")
        for gb in gt_boxes:
            print(f"     {gb}")

        if not result.detections:
            print("   ⚠️  예측: 탐지 없음")
            continue

        print(f"   예측 ({len(result.detections)}개):")
        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            pred_box = [x1, y1, x2, y2]
            best_iou = max((_iou(pred_box, gb) for gb in gt_boxes), default=0.0)
            hit = "✅" if best_iou >= 0.3 else "❌"
            if best_iou >= 0.3:
                hit_count += 1
            print(f"     {hit}  class={det.class_name}  conf={det.confidence:.3f}"
                  f"  bbox={pred_box}  IoU_best={best_iou:.3f}")

    print(f"\n{'='*70}")
    print(f"IoU≥0.3 히트 건수: {hit_count}")


def test_predict_with_resize(plant_svc):
    """이미지를 학습 시 일반적인 크기로 리사이즈 후 추론 — 해상도 불일치 확인."""
    pairs = _collect_samples(max_n=3)
    if not pairs:
        pytest.skip("샘플 이미지 없음")

    import torch
    from server.services.plant.plant_service import DEVICE

    SIZES = [640, 800, 1024, 1333]  # Faster R-CNN 학습 시 자주 쓰는 크기

    print(f"\n{'='*70}")
    print("리사이즈별 탐지 결과 비교")
    print(f"{'='*70}")

    for size in SIZES:
        total_det = 0
        for jpg_path, _ in pairs:
            frame = cv2.imread(str(jpg_path))
            h, w = frame.shape[:2]
            # 긴 쪽 기준 리사이즈 (비율 유지)
            scale = size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))

            result = plant_svc.detect(resized, conf_threshold=0.1)
            total_det += len(result.detections)

        print(f"  size={size}: 총 탐지 {total_det}개 (3개 이미지 합산, conf≥0.1)")

    assert True  # 항상 통과 (진단용)


def test_predict_conf_sweep(plant_svc):
    """conf threshold를 극단적으로 낮춰 raw score 분포 확인."""
    pairs = _collect_samples(max_n=1)
    if not pairs:
        pytest.skip("샘플 이미지 없음")

    import torch
    from server.services.plant.plant_service import DEVICE

    jpg_path, _ = pairs[0]
    frame = cv2.imread(str(jpg_path))
    # 640 크기로 리사이즈
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(DEVICE)

    with torch.no_grad():
        outputs = plant_svc._model([img_tensor])

    raw = outputs[0]
    scores = raw["scores"].cpu().numpy()

    print(f"\n📊 conf sweep (리사이즈 640) — {jpg_path.name}")
    print(f"  전체 proposal 수: {len(scores)}")
    if len(scores) > 0:
        print(f"  max conf : {scores.max():.4f}")
        print(f"  mean conf: {scores.mean():.4f}")
        print(f"  conf 분포:")
        for thr in [0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]:
            cnt = (scores >= thr).sum()
            print(f"    ≥{thr:.2f}: {cnt}개")
    else:
        print("  ⚠️  proposal 0개 — 모델이 아무 박스도 생성하지 않음")

    assert True
    """첫 번째 샘플로 raw 모델 출력(소프트맥스 전)을 상세 출력한다."""
    pairs = _collect_samples(max_n=1)
    if not pairs:
        pytest.skip("샘플 이미지 없음")

    jpg_path, json_path = pairs[0]
    frame = cv2.imread(str(jpg_path))

    import torch
    from server.services.plant.plant_service import DEVICE

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(DEVICE)

    with torch.no_grad():
        outputs = plant_svc._model([img_tensor])

    raw = outputs[0]
    print(f"\n📊 Raw 모델 출력 — {jpg_path.name}")
    print(f"  boxes shape : {raw['boxes'].shape}")
    print(f"  scores shape: {raw['scores'].shape}")
    print(f"  labels shape: {raw['labels'].shape}")

    # conf 기준 내림차순 상위 10개
    scores = raw["scores"].cpu()
    top_k = min(10, len(scores))
    idxs = scores.argsort(descending=True)[:top_k]

    print(f"\n  상위 {top_k}개 예측 (conf 내림차순):")
    for i in idxs:
        s = float(scores[i])
        lbl = int(raw["labels"][i].cpu())
        box = [int(v) for v in raw["boxes"][i].cpu().tolist()]
        cls_name = plant_svc._class_names.get(lbl, f"id={lbl}")
        print(f"    conf={s:.4f}  label={lbl}({cls_name})  box={box}")

    assert raw["boxes"].shape[0] >= 0  # 항상 통과
