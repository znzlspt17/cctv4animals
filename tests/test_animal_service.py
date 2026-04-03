"""AnimalService 단위 테스트 — ultralytics.YOLO 를 mock 하여 순수 로직 검증."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import server.services.animal_service as animal_module
from server.services.animal_service import (
    AnimalDetectionResult,
    AnimalService,
    Detection,
    _parse_results,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_box(conf: float, cls_id: int, xyxy: list[float]) -> MagicMock:
    """ultralytics box mock 생성. box.xyxy[0].tolist() 형태 호출을 지원한다."""
    box = MagicMock()
    box.conf = [conf]
    box.cls = [cls_id]
    xyxy_mock = MagicMock()
    xyxy_mock.tolist.return_value = xyxy
    box.xyxy = [xyxy_mock]
    return box


def _make_mock_result(boxes, names: dict) -> MagicMock:
    """ultralytics Result mock 생성."""
    r = MagicMock()
    r.boxes = boxes
    r.names = names
    return r


def _make_mock_ultralytics(yolo_instance: MagicMock | None = None) -> tuple[MagicMock, MagicMock]:
    """(mock_ultralytics_module, mock_yolo_cls) 튜플 반환."""
    if yolo_instance is None:
        yolo_instance = MagicMock()
    mock_yolo_cls = MagicMock(return_value=yolo_instance)
    mock_module = MagicMock()
    mock_module.YOLO = mock_yolo_cls
    return mock_module, mock_yolo_cls


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def svc() -> AnimalService:
    """깨끗한 AnimalService 인스턴스."""
    return AnimalService()


@pytest.fixture(autouse=True)
def reset_animal_classes():
    """각 테스트 후 ANIMAL_CLASSES를 원래 상태로 복원."""
    original = animal_module.ANIMAL_CLASSES.copy()
    yield
    animal_module.ANIMAL_CLASSES.clear()
    animal_module.ANIMAL_CLASSES.update(original)


# ── _parse_results ────────────────────────────────────────────────────────────


def test_parse_results_empty():
    """boxes=None인 결과를 넣으면 빈 리스트 반환."""
    result = _make_mock_result(boxes=None, names={})
    detections = _parse_results([result], conf_threshold=0.4)
    assert detections == []


def test_parse_results_below_threshold():
    """confidence < threshold인 box는 무시."""
    box = _make_mock_box(conf=0.2, cls_id=0, xyxy=[0.0, 0.0, 10.0, 10.0])
    result = _make_mock_result(boxes=[box], names={0: "cat"})
    detections = _parse_results([result], conf_threshold=0.4)
    assert detections == []


def test_parse_results_single_detection():
    """정상 box 하나 파싱 → Detection 필드 검증."""
    box = _make_mock_box(conf=0.8, cls_id=1, xyxy=[10.0, 20.0, 50.0, 80.0])
    result = _make_mock_result(boxes=[box], names={1: "dog"})
    detections = _parse_results([result], conf_threshold=0.4)

    assert len(detections) == 1
    d = detections[0]
    assert d.class_name == "dog"
    assert abs(d.confidence - 0.8) < 1e-6
    assert d.bbox == [10.0, 20.0, 50.0, 80.0]


def test_parse_results_class_filter():
    """ANIMAL_CLASSES가 설정된 경우 해당 클래스 외 무시."""
    animal_module.ANIMAL_CLASSES.add("cat")
    box_cat = _make_mock_box(conf=0.9, cls_id=0, xyxy=[0.0, 0.0, 10.0, 10.0])
    box_dog = _make_mock_box(conf=0.9, cls_id=1, xyxy=[5.0, 5.0, 20.0, 20.0])
    result = _make_mock_result(boxes=[box_cat, box_dog], names={0: "cat", 1: "dog"})

    detections = _parse_results([result], conf_threshold=0.4)

    assert len(detections) == 1
    assert detections[0].class_name == "cat"


# ── AnimalDetectionResult ─────────────────────────────────────────────────────


def test_result_has_animal_true():
    """detections 있으면 has_animal=True."""
    det = Detection(class_name="cat", confidence=0.9, bbox=[0.0, 0.0, 10.0, 10.0])
    result = AnimalDetectionResult(detections=[det])
    assert result.has_animal is True


def test_result_has_animal_false():
    """detections 없으면 has_animal=False."""
    result = AnimalDetectionResult(detections=[])
    assert result.has_animal is False


def test_result_len():
    """__len__ == len(detections)."""
    detections = [
        Detection(class_name="cat", confidence=0.9, bbox=[0.0, 0.0, 10.0, 10.0]),
        Detection(class_name="dog", confidence=0.8, bbox=[5.0, 5.0, 20.0, 20.0]),
    ]
    result = AnimalDetectionResult(detections=detections)
    assert len(result) == 2


# ── AnimalService.load() ──────────────────────────────────────────────────────


def test_load_file_not_found(svc):
    """MODEL_PATH 없으면 FileNotFoundError."""
    mock_path = MagicMock()
    mock_path.exists.return_value = False
    with patch("server.services.animal_service.MODEL_PATH", mock_path):
        with pytest.raises(FileNotFoundError):
            svc.load()


def test_load_success(svc):
    """MODEL_PATH 있고 YOLO mock → _loaded=True, _model 설정."""
    mock_yolo_instance = MagicMock()
    mock_ultralytics, mock_yolo_cls = _make_mock_ultralytics(mock_yolo_instance)
    mock_path = MagicMock()
    mock_path.exists.return_value = True

    with patch("server.services.animal_service.MODEL_PATH", mock_path):
        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            svc.load()

    assert svc._loaded is True
    assert svc._model is mock_yolo_instance


def test_load_idempotent(svc):
    """두 번 호출해도 YOLO 생성자는 한 번만."""
    mock_ultralytics, mock_yolo_cls = _make_mock_ultralytics()
    mock_path = MagicMock()
    mock_path.exists.return_value = True

    with patch("server.services.animal_service.MODEL_PATH", mock_path):
        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            svc.load()
            svc.load()  # second call — should be no-op

    assert mock_yolo_cls.call_count == 1


# ── AnimalService.detect() ────────────────────────────────────────────────────


def test_detect_calls_model(svc):
    """_loaded 상태에서 detect() → self._model(...) 호출 확인."""
    mock_model = MagicMock(return_value=[])
    svc._model = mock_model
    svc._loaded = True

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    svc.detect(frame, conf_threshold=0.5)

    mock_model.assert_called_once_with(frame, conf=0.5, verbose=False)


def test_detect_returns_result(svc):
    """mock model이 결과 반환 → AnimalDetectionResult 반환."""
    box = _make_mock_box(conf=0.9, cls_id=0, xyxy=[0.0, 0.0, 10.0, 10.0])
    mock_result = _make_mock_result(boxes=[box], names={0: "cat"})
    svc._model = MagicMock(return_value=[mock_result])
    svc._loaded = True

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = svc.detect(frame)

    assert isinstance(result, AnimalDetectionResult)
    assert result.has_animal is True
    assert result.detections[0].class_name == "cat"


def test_detect_triggers_load_if_not_loaded(svc):
    """_loaded=False일 때 detect() → load() 자동 호출."""
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_model = MagicMock(return_value=[])

    def fake_load():
        svc._loaded = True
        svc._model = mock_model

    with patch.object(svc, "load", side_effect=fake_load) as mock_load:
        svc.detect(frame)

    mock_load.assert_called_once()


# ── AnimalService.is_ready() / warmup() ──────────────────────────────────────


def test_is_ready_before_load(svc):
    """초기 상태 False."""
    assert svc.is_ready() is False


def test_is_ready_after_load(svc):
    """load() 후 True."""
    mock_ultralytics, _ = _make_mock_ultralytics()
    mock_path = MagicMock()
    mock_path.exists.return_value = True

    with patch("server.services.animal_service.MODEL_PATH", mock_path):
        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            svc.load()

    assert svc.is_ready() is True


def test_warmup_calls_model(svc):
    """warmup() → 640x640 더미 이미지로 model 호출 확인."""
    mock_model = MagicMock(return_value=[])
    svc._model = mock_model
    svc._loaded = True

    svc.warmup()

    assert mock_model.called
    call_args = mock_model.call_args
    dummy_frame = call_args[0][0]
    assert dummy_frame.shape == (640, 640, 3)
    assert call_args[1].get("verbose") is False
