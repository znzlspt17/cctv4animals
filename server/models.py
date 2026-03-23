from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from server.database import Base


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    phone = Column(String(50), nullable=True)
    address = Column(Text, nullable=True)
    extra_info = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    face_images = relationship(
        "FaceImage", back_populates="person", cascade="all, delete-orphan"
    )
    recognition_logs = relationship("RecognitionLog", back_populates="person")
    alert_rules = relationship(
        "AlertRule", back_populates="person", cascade="all, delete-orphan"
    )


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
    created_at = Column(DateTime, default=func.now())

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
    recognized_at = Column(DateTime, default=func.now(), index=True)

    person = relationship("Person", back_populates="recognition_logs")

    __table_args__ = (
        Index("ix_recognition_logs_person_recognized", "person_id", "recognized_at"),
    )


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(
        Integer,
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    alert_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

    person = relationship("Person", back_populates="alert_rules")


class PersonNameSeq(Base):
    __tablename__ = "person_name_seq"

    id = Column(Integer, primary_key=True, default=1)
    next_val = Column(Integer, nullable=False, default=1)
