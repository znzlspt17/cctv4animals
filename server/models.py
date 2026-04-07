from datetime import datetime, timedelta, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from server.database import Base

_KST = timezone(timedelta(hours=9))


def _now_kst() -> datetime:
    return datetime.now(_KST)


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    phone = Column(String(50), nullable=True)
    address = Column(Text, nullable=True)
    extra_info = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now_kst)
    updated_at = Column(DateTime(timezone=True), default=_now_kst, onupdate=_now_kst)

    face_images = relationship(
        "FaceImage", back_populates="person", cascade="all, delete-orphan"
    )
    recognition_logs = relationship("RecognitionLog", back_populates="person")


class FaceImage(Base):
    __tablename__ = "face_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(
        Integer,
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_path = Column(String(500), nullable=False)
    capture_condition = Column(String(100), nullable=True)
    embedding = Column(LargeBinary, nullable=True)
    embedding_vec = Column(Vector(512), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now_kst)

    person = relationship("Person", back_populates="face_images")


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    person_id = Column(
        Integer,
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    confidence = Column(Float, nullable=False)
    snapshot_path = Column(String(500), nullable=True)
    recognized_at = Column(DateTime(timezone=True), default=_now_kst, index=True)

    person = relationship("Person", back_populates="recognition_logs")

    __table_args__ = (
        Index("ix_recognition_logs_person_recognized", "person_id", "recognized_at"),
    )



class PersonNameSeq(Base):
    __tablename__ = "person_name_seq"

    id = Column(Integer, primary_key=True, default=1)
    next_val = Column(Integer, nullable=False, default=1)


class TrackingEvent(Base):
    __tablename__ = "tracking_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), nullable=False, index=True)
    tracker_id = Column(Integer, nullable=False)
    direction = Column(Enum("IN", "OUT", name="direction_enum"), nullable=False)
    count_change = Column(Integer, nullable=False)
    current_count = Column(Integer, nullable=False, default=0)
    confidence = Column(Float, nullable=True)
    bbox_x = Column(Integer, nullable=True)
    bbox_y = Column(Integer, nullable=True)
    bbox_w = Column(Integer, nullable=True)
    bbox_h = Column(Integer, nullable=True)
    snapshot_path = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now_kst, index=True)


class AnimalDetectionLog(Base):
    """동물 탐지 추론 결과 로그."""

    __tablename__ = "animal_detection_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False, default="api")   # "api" | "webcam" | "video"
    class_name = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    image_data = Column(LargeBinary, nullable=True)
    detected_at = Column(DateTime(timezone=True), default=_now_kst, index=True)

    __table_args__ = (
        Index("ix_animal_logs_class_at", "class_name", "detected_at"),
    )


class PlantDetectionLog(Base):
    """식물 탐지 추론 결과 로그."""

    __tablename__ = "plant_detection_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False, default="api")
    class_name = Column(String(100), nullable=False, index=True)
    disease_code = Column(Integer, nullable=True)
    disease_label = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    # 표2 메타
    crop_type = Column(Integer, nullable=True)
    crop_name = Column(String(100), nullable=True)
    shooting_type = Column(Integer, nullable=True)
    shooting_type_name = Column(String(50), nullable=True)
    grow_stage = Column(Integer, nullable=True)
    grow_stage_name = Column(String(50), nullable=True)
    area = Column(Integer, nullable=True)
    area_name = Column(String(50), nullable=True)
    image_data = Column(LargeBinary, nullable=True)
    detected_at = Column(DateTime(timezone=True), default=_now_kst, index=True)

    __table_args__ = (
        Index("ix_plant_logs_class_at", "class_name", "detected_at"),
    )
