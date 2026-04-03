import logging
from datetime import datetime, timedelta

from sqlalchemy import func

from server.database import SessionLocal
from server.models import AlertRule, FaceImage, Person, PersonNameSeq, RecognitionLog
from server.repositories.base import (
    AbstractRepository,
    AlertRuleRepo,
    FaceImageRepo,
    PersonRepo,
    RecognitionLogRepo,
    SeqRepo,
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


class _PostgresAlertRuleRepo(AlertRuleRepo):
    def get_active_rules(self, person_id: int):
        with SessionLocal() as db:
            return (
                db.query(AlertRule)
                .filter(
                    AlertRule.person_id == person_id,
                    AlertRule.is_active.is_(True),
                )
                .all()
            )

    def upsert(
        self,
        person_id: int,
        alert_type: str,
        message: str,
        is_active: bool = True,
    ):
        with SessionLocal() as db:
            rule = (
                db.query(AlertRule)
                .filter(
                    AlertRule.person_id == person_id,
                    AlertRule.alert_type == alert_type,
                )
                .first()
            )
            if rule:
                rule.message = message
                rule.is_active = is_active
            else:
                rule = AlertRule(
                    person_id=person_id,
                    alert_type=alert_type,
                    message=message,
                    is_active=is_active,
                )
                db.add(rule)
            db.commit()
            db.refresh(rule)
            return rule

    def get_by_person(self, person_id: int):
        with SessionLocal() as db:
            return db.query(AlertRule).filter(AlertRule.person_id == person_id).all()

    def delete(self, rule_id: int) -> None:
        with SessionLocal() as db:
            rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
            if rule:
                db.delete(rule)
                db.commit()


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


class PostgresRepository(AbstractRepository):
    def __init__(self):
        self._person = _PostgresPersonRepo()
        self._face_image = _PostgresFaceImageRepo()
        self._recognition_log = _PostgresRecognitionLogRepo()
        self._alert_rule = _PostgresAlertRuleRepo()
        self._seq = _PostgresSeqRepo()

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
    def alert_rule(self) -> _PostgresAlertRuleRepo:
        return self._alert_rule

    @property
    def seq(self) -> _PostgresSeqRepo:
        return self._seq


# 하위 호환성 별칭
MySQLRepository = PostgresRepository
