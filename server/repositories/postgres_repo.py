import logging
from datetime import datetime, timedelta

from sqlalchemy import func

from server.database import SessionLocal
from server.models import FaceImage, Person, PersonNameSeq, RecognitionLog, TrackingEvent, AnimalDetectionLog, PlantDetectionLog
from server.repositories.base import (
    AbstractRepository,
    AnimalDetectionLogRepo,
    FaceImageRepo,
    PersonRepo,
    PlantDetectionLogRepo,
    RecognitionLogRepo,
    SeqRepo,
    TrackingEventRepo,
)

logger = logging.getLogger(__name__)


# ── Sub-repo implementations ──


class _PostgresPersonRepo(PersonRepo):
    def create(self, name: str, **kwargs):
        with SessionLocal() as db:
            person = Person(name=name, **kwargs)
            db.add(person)
            db.commit()
            db.refresh(person)
            return person

    def get(self, person_id: int):
        with SessionLocal() as db:
            return db.query(Person).filter(Person.id == person_id).first()

    def list_all(self):
        with SessionLocal() as db:
            return db.query(Person).order_by(Person.id).all()

    def update(self, person_id: int, **kwargs):
        with SessionLocal() as db:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                return None
            for key, value in kwargs.items():
                if hasattr(person, key):
                    setattr(person, key, value)
            db.commit()
            db.refresh(person)
            return person

    def delete(self, person_id: int) -> None:
        with SessionLocal() as db:
            person = db.query(Person).filter(Person.id == person_id).first()
            if person:
                db.delete(person)
                db.commit()


class _PostgresFaceImageRepo(FaceImageRepo):
    def create(
        self,
        person_id: int,
        image_path: str,
        embedding: bytes | None = None,
        embedding_vec: list | None = None,
        capture_condition: str | None = None,
    ):
        with SessionLocal() as db:
            face = FaceImage(
                person_id=person_id,
                image_path=image_path,
                embedding=embedding,
                embedding_vec=embedding_vec,
                capture_condition=capture_condition,
            )
            db.add(face)
            db.commit()
            db.refresh(face)
            return face

    def get_by_person(self, person_id: int):
        with SessionLocal() as db:
            return db.query(FaceImage).filter(FaceImage.person_id == person_id).all()

    def get_all_embeddings(self):
        with SessionLocal() as db:
            rows = (
                db.query(FaceImage.person_id, FaceImage.embedding)
                .filter(FaceImage.embedding.isnot(None))
                .all()
            )
            return [
                {"person_id": row.person_id, "embedding": row.embedding} for row in rows
            ]

    def delete_by_person(self, person_id: int) -> None:
        with SessionLocal() as db:
            db.query(FaceImage).filter(FaceImage.person_id == person_id).delete()
            db.commit()


class _PostgresRecognitionLogRepo(RecognitionLogRepo):
    def create(
        self,
        person_id: int | None,
        confidence: float,
        snapshot_path: str | None = None,
    ):
        with SessionLocal() as db:
            log = RecognitionLog(
                person_id=person_id,
                confidence=confidence,
                snapshot_path=snapshot_path,
            )
            db.add(log)
            db.commit()
            db.refresh(log)
            return log

    def query(
        self,
        start_date=None,
        end_date=None,
        person_id=None,
        page: int = 1,
        page_size: int = 20,
    ):
        with SessionLocal() as db:
            q = db.query(RecognitionLog)
            if start_date:
                q = q.filter(RecognitionLog.recognized_at >= start_date)
            if end_date:
                q = q.filter(RecognitionLog.recognized_at <= end_date)
            if person_id is not None:
                q = q.filter(RecognitionLog.person_id == person_id)
            q = q.order_by(RecognitionLog.recognized_at.desc())
            offset = (page - 1) * page_size
            return q.offset(offset).limit(page_size).all()

    def get_stats(self):
        with SessionLocal() as db:
            total = db.query(func.count(RecognitionLog.id)).scalar() or 0
            rows = (
                db.query(
                    RecognitionLog.person_id,
                    Person.name,
                    Person.display_name,
                    func.count(RecognitionLog.id).label("count"),
                )
                .outerjoin(Person, RecognitionLog.person_id == Person.id)
                .group_by(RecognitionLog.person_id, Person.name, Person.display_name)
                .all()
            )
            person_stats = [
                {
                    "person_id": row.person_id,
                    "person_name": row.name,
                    "display_name": row.display_name,
                    "count": row.count,
                }
                for row in rows
            ]
            return {"total_logs": total, "person_stats": person_stats}

    def cleanup(self, retention_days: int) -> int:
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        with SessionLocal() as db:
            count = (
                db.query(RecognitionLog)
                .filter(RecognitionLog.recognized_at < cutoff)
                .delete()
            )
            db.commit()
            return count

    def is_duplicate_log(self, person_id: int, dedup_seconds: int) -> bool:
        cutoff = datetime.utcnow() - timedelta(seconds=dedup_seconds)
        with SessionLocal() as db:
            exists = (
                db.query(RecognitionLog)
                .filter(
                    RecognitionLog.person_id == person_id,
                    RecognitionLog.recognized_at >= cutoff,
                )
                .first()
            )
            return exists is not None


class _PostgresSeqRepo(SeqRepo):
    def next_person_number(self) -> int:
        with SessionLocal() as db:
            row = (
                db.query(PersonNameSeq)
                .filter(PersonNameSeq.id == 1)
                .with_for_update()
                .first()
            )
            if not row:
                row = PersonNameSeq(id=1, next_val=1)
                db.add(row)
                db.flush()
            val = row.next_val
            row.next_val = val + 1
            db.commit()
            return val


# ── Composite Repository ──


class _PostgresTrackingEventRepo(TrackingEventRepo):
    def save(
        self,
        camera_id: str,
        tracker_id: int,
        direction: str,
        count_change: int,
        current_count: int,
        confidence: float,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
        snapshot_path: str = "",
    ):
        with SessionLocal() as db:
            event = TrackingEvent(
                camera_id=camera_id,
                tracker_id=tracker_id,
                direction=direction,
                count_change=count_change,
                current_count=current_count,
                confidence=confidence,
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_w=bbox_w,
                bbox_h=bbox_h,
                snapshot_path=snapshot_path,
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            return event

    def get_current_count(self, camera_id: str) -> int:
        with SessionLocal() as db:
            row = (
                db.query(TrackingEvent.current_count)
                .filter(TrackingEvent.camera_id == camera_id)
                .order_by(TrackingEvent.id.desc())
                .first()
            )
            return row[0] if row else 0

    def get_events(self, camera_id: str, limit: int = 100) -> list[dict]:
        with SessionLocal() as db:
            rows = (
                db.query(TrackingEvent)
                .filter(TrackingEvent.camera_id == camera_id)
                .order_by(TrackingEvent.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "camera_id": r.camera_id,
                    "tracker_id": r.tracker_id,
                    "direction": r.direction,
                    "count_change": r.count_change,
                    "current_count": r.current_count,
                    "confidence": r.confidence,
                    "snapshot_path": r.snapshot_path,
                    "created_at": r.created_at,
                }
                for r in rows
            ]


class _PostgresAnimalDetectionLogRepo(AnimalDetectionLogRepo):
    def create(
        self,
        class_name: str,
        confidence: float,
        bbox: list[float],
        source: str = "api",
    ):
        x1, y1, x2, y2 = (bbox + [None, None, None, None])[:4]
        with SessionLocal() as db:
            log = AnimalDetectionLog(
                source=source,
                class_name=class_name,
                confidence=confidence,
                bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
            )
            db.add(log)
            db.commit()
            db.refresh(log)
            return log

    def query(self, limit: int = 100) -> list:
        with SessionLocal() as db:
            return (
                db.query(AnimalDetectionLog)
                .order_by(AnimalDetectionLog.detected_at.desc())
                .limit(limit)
                .all()
            )


class _PostgresPlantDetectionLogRepo(PlantDetectionLogRepo):
    def create(
        self,
        class_name: str,
        confidence: float,
        bbox: list[float],
        disease_code: int | None = None,
        disease_label: str | None = None,
        source: str = "api",
        crop_type: int | None = None,
        crop_name: str | None = None,
        shooting_type: int | None = None,
        shooting_type_name: str | None = None,
        grow_stage: int | None = None,
        grow_stage_name: str | None = None,
        area: int | None = None,
        area_name: str | None = None,
    ):
        x1, y1, x2, y2 = (bbox + [None, None, None, None])[:4]
        with SessionLocal() as db:
            log = PlantDetectionLog(
                source=source,
                class_name=class_name,
                disease_code=disease_code,
                disease_label=disease_label,
                confidence=confidence,
                bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                crop_type=crop_type, crop_name=crop_name,
                shooting_type=shooting_type, shooting_type_name=shooting_type_name,
                grow_stage=grow_stage, grow_stage_name=grow_stage_name,
                area=area, area_name=area_name,
            )
            db.add(log)
            db.commit()
            db.refresh(log)
            return log

    def query(self, limit: int = 100) -> list:
        with SessionLocal() as db:
            return (
                db.query(PlantDetectionLog)
                .order_by(PlantDetectionLog.detected_at.desc())
                .limit(limit)
                .all()
            )


class PostgresRepository(AbstractRepository):
    def __init__(self):
        self._person = _PostgresPersonRepo()
        self._face_image = _PostgresFaceImageRepo()
        self._recognition_log = _PostgresRecognitionLogRepo()
        self._seq = _PostgresSeqRepo()
        self._tracking_event = _PostgresTrackingEventRepo()
        self._animal_detection_log = _PostgresAnimalDetectionLogRepo()
        self._plant_detection_log = _PostgresPlantDetectionLogRepo()

    @property
    def person(self) -> _PostgresPersonRepo:
        return self._person

    @property
    def face_image(self) -> _PostgresFaceImageRepo:
        return self._face_image

    @property
    def recognition_log(self) -> _PostgresRecognitionLogRepo:
        return self._recognition_log

    @property
    def seq(self) -> _PostgresSeqRepo:
        return self._seq

    @property
    def tracking_event(self) -> _PostgresTrackingEventRepo:
        return self._tracking_event

    @property
    def animal_detection_log(self) -> _PostgresAnimalDetectionLogRepo:
        return self._animal_detection_log

    @property
    def plant_detection_log(self) -> _PostgresPlantDetectionLogRepo:
        return self._plant_detection_log
