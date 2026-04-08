import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "best_model_lettuce.pt"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 상추 crop_type 코드 (plant_service.py CROP_NAMES 기준)
LETTUCE_CROP_TYPE = 11
LETTUCE_CROP_NAME = "상추"

# 질병 코드 한글 명칭 (식물 공통 코드계)
DISEASE_LABELS: dict[int, str] = {
    0: "정상",
    1: "잿빛곰팡이병",
    2: "균핵병",
    3: "노균병",
    4: "무름병",
    5: "위조병",
    6: "탄저병",
    7: "회색곰팡이병",
    8: "흰가루병",
    9: "시들음병",
    10: "역병",
}


@dataclass
class LettuceDetection:
    class_name: str
    confidence: float
    bbox: list[float]          # [x1, y1, x2, y2]
    disease_code: int = 0
    disease_label: str = ""


@dataclass
class LettuceDetectionResult:
    detections: list[LettuceDetection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.detections)

    @property
    def has_lettuce(self) -> bool:
        return len(self.detections) > 0


def _parse_output(
    output: dict,
    label_to_name: dict,
    label_to_disease: dict,
    conf_threshold: float,
) -> list[LettuceDetection]:
    detections: list[LettuceDetection] = []
    boxes = output["boxes"].cpu()
    scores = output["scores"].cpu()
    labels = output["labels"].cpu()

    for box, score, label in zip(boxes, scores, labels):
        conf = float(score)
        if conf < conf_threshold:
            continue
        cls_id = int(label)
        class_name = label_to_name.get(cls_id, str(cls_id))
        d_code = label_to_disease.get(cls_id, 0)
        d_label = DISEASE_LABELS.get(d_code, class_name)
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            LettuceDetection(
                class_name=class_name,
                confidence=conf,
                bbox=[x1, y1, x2, y2],
                disease_code=d_code,
                disease_label=d_label,
            )
        )

    return detections


class LettuceService:
    """Faster R-CNN (ConvNeXt) 기반 상추 탐지 서비스."""

    def __init__(self):
        self._model: torch.nn.Module | None = None
        self._label_to_name: dict[int, str] = {}
        self._label_to_disease: dict[int, int] = {}  # label_index → disease_code
        self._loaded: bool = False

    def load(self) -> None:
        if self._loaded:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Lettuce model not found: {MODEL_PATH}\n"
                "models/ 디렉토리에 best_model_lettuce.pt 파일을 배치하세요."
            )

        from lettuce_checkpoint_loader import load_detection_checkpoint

        loaded = load_detection_checkpoint(
            model_path=MODEL_PATH,
            device=torch.device(DEVICE),
        )
        self._model = loaded["model"]
        self._label_to_name = loaded["label_to_name"]
        # disease_to_label: {disease_code: label_index} → invert to {label_index: disease_code}
        disease_to_label: dict[int, int] = loaded["disease_to_label"]
        self._label_to_disease = {v: k for k, v in disease_to_label.items()}

        self._model.eval()
        self._loaded = True
        logger.info(
            "LettuceService: model loaded from %s (device=%s, classes=%s)",
            MODEL_PATH, DEVICE, self._label_to_name,
        )

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> LettuceDetectionResult:
        if not self._loaded:
            self.load()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        img_tensor = img_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = self._model([img_tensor])

        detections = _parse_output(
            outputs[0],
            self._label_to_name,
            self._label_to_disease,
            conf_threshold,
        )
        return LettuceDetectionResult(detections=detections)

    def is_ready(self) -> bool:
        return self._loaded

    def warmup(self) -> None:
        if not self._loaded:
            self.load()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, conf_threshold=0.1)
        logger.info("LettuceService: warmup complete")

    @property
    def class_names(self) -> dict[int, str]:
        return self._label_to_name


# 싱글톤 인스턴스
lettuce_service = LettuceService()
