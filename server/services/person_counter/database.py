"""SQLAlchemy models and database connection module."""

from __future__ import annotations

from datetime import datetime

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import settings

# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------

_DB_URL = (
    f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?sslmode=disable"
)

engine = create_engine(
    _DB_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------


class TrackingEvent(Base):  # type: ignore[misc]
    """tracking_events table – mirrors §5.1 schema."""

    __tablename__ = "tracking_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, nullable=False)
    camera_id = Column(String(50), nullable=False)
    direction = Column(Enum("IN", "OUT", name="direction_enum"), nullable=False)
    count_change = Column(Integer, nullable=False)
    current_count = Column(Integer, nullable=False, default=0)
    confidence = Column(Float, nullable=True)
    bbox_x = Column(Integer, nullable=True)
    bbox_y = Column(Integer, nullable=True)
    bbox_w = Column(Integer, nullable=True)
    bbox_h = Column(Integer, nullable=True)
    tracking_start = Column(DateTime, nullable=True)
    tracking_end = Column(DateTime, nullable=True)
    snapshot_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=func.now())


# ---------------------------------------------------------------------------
# DB Initialization
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create all tables (PostgreSQL DB must already exist)."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("All tables created / verified.")
    except Exception as exc:
        logger.error("Failed to create tables: {}", exc)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_db():
    """FastAPI Depends generator that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# CRUD Functions
# ---------------------------------------------------------------------------


def save_event(db: Session, event_data: dict) -> TrackingEvent:
    """Insert a new tracking event and return the ORM instance."""
    event = TrackingEvent(**event_data)
    db.add(event)
    db.commit()
    db.refresh(event)
    logger.debug("Saved event id={} direction={}", event.id, event.direction)
    return event


def get_current_count(db: Session, camera_id: str) -> int:
    """Return the most recent current_count for *camera_id*, or 0."""
    row = (
        db.query(TrackingEvent.current_count)
        .filter(TrackingEvent.camera_id == camera_id)
        .order_by(TrackingEvent.id.desc())
        .first()
    )
    return row[0] if row else 0


def get_events(
    db: Session,
    camera_id: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 100,
) -> list[TrackingEvent]:
    """Query tracking events with optional time-range filter."""
    q = db.query(TrackingEvent).filter(TrackingEvent.camera_id == camera_id)
    if start_time:
        q = q.filter(TrackingEvent.created_at >= start_time)
    if end_time:
        q = q.filter(TrackingEvent.created_at <= end_time)
    return q.order_by(TrackingEvent.id.desc()).limit(limit).all()


# ---------------------------------------------------------------------------
# State Recovery (Step 2B.4)
# ---------------------------------------------------------------------------


def get_last_state(db: Session, camera_id: str) -> dict:
    """Return the last persisted state for *camera_id*.

    Returns dict with ``current_count``, ``in_count``, ``out_count``,
    and ``recent_person_ids`` (last 200 tracked IDs to avoid re-counting).
    """
    last_event = (
        db.query(TrackingEvent)
        .filter(TrackingEvent.camera_id == camera_id)
        .order_by(TrackingEvent.id.desc())
        .first()
    )
    if not last_event:
        return {
            "current_count": 0,
            "in_count": 0,
            "out_count": 0,
            "recent_person_ids": [],
        }

    in_count = (
        db.query(func.count(TrackingEvent.id))
        .filter(TrackingEvent.camera_id == camera_id, TrackingEvent.direction == "IN")
        .scalar()
    ) or 0

    out_count = (
        db.query(func.count(TrackingEvent.id))
        .filter(TrackingEvent.camera_id == camera_id, TrackingEvent.direction == "OUT")
        .scalar()
    ) or 0

    recent_ids_rows = (
        db.query(TrackingEvent.person_id)
        .filter(TrackingEvent.camera_id == camera_id)
        .order_by(TrackingEvent.id.desc())
        .limit(200)
        .all()
    )
    recent_ids = [r[0] for r in recent_ids_rows]

    return {
        "current_count": last_event.current_count,
        "in_count": in_count,
        "out_count": out_count,
        "recent_person_ids": recent_ids,
    }


def restore_count(camera_id: str) -> int:
    """Convenience: open a session, fetch last current_count, close session."""
    db = SessionLocal()
    try:
        count = get_current_count(db, camera_id)
        logger.info("Restored count for {}: {}", camera_id, count)
        return count
    except Exception as exc:
        logger.warning("Could not restore count for {}: {}", camera_id, exc)
        return 0
    finally:
        db.close()
