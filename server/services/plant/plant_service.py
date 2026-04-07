import logging
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "best-4plants-fasterRCNN.pt"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── 표2 기준 코드 정의 ────────────────────────────────────────────────────────
# 작물 명칭 (crop) 코드
CROP_NAMES: dict[int, str] = {
    0: "딸기",       1: "토마토",   2: "파프리카", 3: "오이",
    4: "딸기(시설)", 5: "고추",    6: "수박",     7: "멜론",
    8: "호박",       9: "가지",    10: "배추",    11: "상추",
}

# 촬영 유형 (type) 코드
SHOOTING_TYPES: dict[int, str] = {
    0: "Normal", 1: "Complaint", 2: "Insect", 3: "Damage",
}

# 작물 생육 단계 (grow) 코드
GROW_STAGES: dict[int, str] = {
    11: "육묘기", 12: "생장기", 13: "착화/과실기",
}

# 촬영 부위 (area) 코드
AREA_TYPES: dict[int, str] = {
    0: "None", 1: "Fruit", 2: "Flower", 3: "Leaf",
    4: "Branch", 5: "Stem", 6: "Root",
}

# 질병 피해 정도 (risk) 코드 — 모델이 예측하지 않으며 어노테이션 메타 값
RISK_LEVELS: dict[int, str] = {
    0: "정상", 1: "초기", 2: "중기", 3: "말기",
}

# 질병 코드 (disease) ↔ 클래스명 매핑
DISEASE_CODE_TO_CLASS: dict[int, str] = {
    0: "normal", 7: "gray_mold", 8: "powdery_mildew",
}
CLASS_TO_DISEASE_CODE: dict[str, int] = {v: k for k, v in DISEASE_CODE_TO_CLASS.items()}

# 질병 코드 한글 명칭
DISEASE_LABELS: dict[int, str] = {
    0: "정상", 7: "회색곰팡이병", 8: "흰가루병",
}


@dataclass
class Detection:
    """단일 객체 탐지 결과."""
    class_name: str
    confidence: float
    bbox: list[float]          # [x1, y1, x2, y2]
    disease_code: int = 0      # 원본 disease 코드 (0/7/8)
    disease_label: str = ""    # 한글 질병명


@dataclass
class PlantDetectionResult:
    """단일 프레임의 식물 탐지 결과."""
    detections: list[Detection] = field(default_factory=list)
    # 호출자가 제공하는 표2 메타데이터 (모델 예측 대상 아님)
    crop_type: int | None = None
    crop_name: str = ""
    shooting_type: int | None = None
    shooting_type_name: str = ""
    grow_stage: int | None = None
    grow_stage_name: str = ""
    area: int | None = None
    area_name: str = ""

    def __len__(self) -> int:
        return len(self.detections)

    @property
    def has_plant(self) -> bool:
        return len(self.detections) > 0


def _build_model(num_classes: int) -> torch.nn.Module:
    """ConvNeXt-Small + FPN 기반 Faster R-CNN 모델 구조 재구성.

    학습 노트북(strawberry_test.ipynb)의 build_convnext_fasterrcnn과 동일한 구조:
    - BackboneWithFPN: return_layers={"2":"0","4":"1","6":"2"} (downsampling 레이어)
    - ROI pooler: featmap_names=["0","1","2","pool"] (LastLevelMaxPool 포함)
    - 4 anchor sizes, min/max size=640
    """
    import torchvision
    from torchvision.models import convnext_small
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.models.detection.rpn import AnchorGenerator

    backbone_body = convnext_small(weights=None).features

    backbone = BackboneWithFPN(
        backbone=backbone_body,
        return_layers={"2": "0", "4": "1", "6": "2"},
        in_channels_list=[192, 384, 768],
        out_channels=256,
    )

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "pool"],
        output_size=7,
        sampling_ratio=2,
    )

    return FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=640,
        max_size=640,
        box_score_thresh=0.001,
    )


def _parse_output(output: dict, names: dict, conf_threshold: float) -> list[Detection]:
    """Faster R-CNN 출력 dict → Detection 리스트 변환."""
    detections: list[Detection] = []
    boxes = output["boxes"].cpu()
    scores = output["scores"].cpu()
    labels = output["labels"].cpu()

    for box, score, label in zip(boxes, scores, labels):
        conf = float(score)
        if conf < conf_threshold:
            continue
        cls_id = int(label)
        class_name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = box.tolist()
        d_code = CLASS_TO_DISEASE_CODE.get(class_name, 0)
        d_label = DISEASE_LABELS.get(d_code, "")
        detections.append(
            Detection(
                class_name=class_name,
                confidence=conf,
                bbox=[x1, y1, x2, y2],
                disease_code=d_code,
                disease_label=d_label,
            )
        )

    return detections


class PlantService:
    """Faster R-CNN (ConvNeXt-Small) 기반 식물 탐지 서비스 — 싱글톤으로 사용."""

    def __init__(self):
        self._model: torch.nn.Module | None = None
        self._class_names: dict = {}
        self._loaded: bool = False

    def load(self) -> None:
        if self._loaded:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Plant model not found: {MODEL_PATH}\n"
                "models/ 디렉토리에 best-4plants-fasterRCNN.pt 파일을 배치하세요."
            )
        ckpt = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
        num_classes: int = ckpt["num_classes"]
        self._class_names = ckpt.get("class_names", {})

        model = _build_model(num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        self._model = model
        self._loaded = True
        logger.info(
            "PlantService: model loaded from %s (device=%s, classes=%s)",
            MODEL_PATH, DEVICE, self._class_names,
        )

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.4,
        crop_type: int | None = None,
        shooting_type: int | None = None,
        grow_stage: int | None = None,
        area: int | None = None,
    ) -> PlantDetectionResult:
        """BGR 프레임에서 식물을 탐지하고 결과를 반환한다."""
        if not self._loaded:
            self.load()

        # BGR → RGB float32 [0, 1] tensor
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        img_tensor = img_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = self._model([img_tensor])

        detections = _parse_output(outputs[0], self._class_names, conf_threshold)
        return PlantDetectionResult(
            detections=detections,
            crop_type=crop_type,
            crop_name=CROP_NAMES.get(crop_type, "") if crop_type is not None else "",
            shooting_type=shooting_type,
            shooting_type_name=SHOOTING_TYPES.get(shooting_type, "") if shooting_type is not None else "",
            grow_stage=grow_stage,
            grow_stage_name=GROW_STAGES.get(grow_stage, "") if grow_stage is not None else "",
            area=area,
            area_name=AREA_TYPES.get(area, "") if area is not None else "",
        )

    def is_ready(self) -> bool:
        return self._loaded

    def warmup(self) -> None:
        if not self._loaded:
            self.load()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, conf_threshold=0.1)
        logger.info("PlantService: warmup complete")


# 싱글톤 인스턴스
plant_service = PlantService()
