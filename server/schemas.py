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


class AlertRuleCreate(BaseModel):
    person_id: int
    alert_type: str = "toast"
    message: str
    is_active: bool = True


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
    created_at: datetime
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
    alerts: list[dict] = []


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

    model_config = {"from_attributes": True}


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
