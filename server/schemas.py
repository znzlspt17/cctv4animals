from datetime import datetime

from pydantic import BaseModel

# ── Request Schemas ──


class PersonCreate(BaseModel):
    display_name: str | None = None
    phone: str | None = None
    address: str | None = None
    extra_info: dict | None = None


class PersonUpdate(BaseModel):
    display_name: str | None = None
    phone: str | None = None
    address: str | None = None
    extra_info: dict | None = None


class LogQueryParams(BaseModel):
    start_date: datetime | None = None
    end_date: datetime | None = None
    person_id: int | None = None
    page: int = 1
    page_size: int = 20


# ── Response Schemas ──


class PersonResponse(BaseModel):
    id: int
    name: str
    display_name: str | None = None
    phone: str | None = None
    address: str | None = None
    extra_info: dict | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class FaceImageResponse(BaseModel):
    id: int
    person_id: int
    image_path: str
    capture_condition: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PersonDetailResponse(PersonResponse):
    face_images: list[FaceImageResponse] = []


class RecognitionResult(BaseModel):
    person_id: int | None = None
    person_name: str | None = None
    display_name: str | None = None
    confidence: float
    bbox: list[int] = []


class RecognizeResponse(BaseModel):
    results: list[RecognitionResult]


class RegisterResponse(BaseModel):
    person_id: int
    person_name: str
    face_image_id: int
    message: str


class LogResponse(BaseModel):
    id: int
    person_id: int | None = None
    person_name: str | None = None
    confidence: float
    snapshot_path: str | None = None
    recognized_at: datetime

    model_config = {"from_attributes": True}


class LogStatsResponse(BaseModel):
    total_logs: int
    person_stats: list[dict]


class AlertResponse(BaseModel):
    id: int
    person_id: int
    alert_type: str
    message: str
    is_active: bool


# ── Camera Schemas ──


class CameraConfigRequest(BaseModel):
    """런타임에 카메라를 동적으로 추가할 때 사용하는 요청 바디."""

    camera_id: str
    video_source: str
    label: str = ""
    line_start_x: int = 0
    line_start_y: int = 360
    line_end_x: int = 1280
    line_end_y: int = 360
    frame_skip: int = 10
    confidence_threshold: float = 0.5
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 0
    roi_h: int = 0
    snapshot_dir: str = "snapshots"


class CameraStatusResponse(BaseModel):
    """카메라 상태 응답."""

    camera_id: str
    label: str
    video_source: str
    current_count: int
    in_count: int
    out_count: int
    running: bool
    paused: bool
    error: str | None = None


class CameraAggregateResponse(BaseModel):
    """전체 카메라 합산 응답."""

    total_current_count: int
    total_in_count: int
    total_out_count: int
    camera_count: int


    model_config = {"from_attributes": True}


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
