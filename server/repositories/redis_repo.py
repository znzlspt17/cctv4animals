import logging
import struct
from datetime import datetime, timedelta

from server.redis_client import get_redis_client
from server.repositories.base import (
    AbstractRepository,
    AlertRuleRepo,
    FaceImageRepo,
    PersonRepo,
    RecognitionLogRepo,
    SeqRepo,
)

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 4096  # VGG-Face


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


class _ObjProxy:
    """Lightweight attribute-access wrapper over a dict (mimics ORM row)."""

    def __init__(self, data: dict):
        self.__dict__.update(data)

    def __repr__(self):
        return f"<ObjProxy {self.__dict__}>"


def _to_obj(data: dict) -> _ObjProxy:
    """Convert Redis JSON dict to attribute-accessible object."""
    obj = dict(data)
    for key in ("created_at", "updated_at", "recognized_at"):
        if key in obj and isinstance(obj[key], str):
            obj[key] = _parse_iso(obj[key])
    return _ObjProxy(obj)


# ── Index initialization ──


def init_redis_indexes(r=None):
    """Create RediSearch indexes if they don't exist."""
    if r is None:
        r = get_redis_client()

    _create_index_if_missing(
        r,
        "idx:persons",
        "person:",
        "SCHEMA",
        "$.id",
        "AS",
        "id",
        "NUMERIC",
        "$.name",
        "AS",
        "name",
        "TAG",
        "$.display_name",
        "AS",
        "display_name",
        "TEXT",
    )
    _create_index_if_missing(
        r,
        "idx:face_images",
        "face:",
        "SCHEMA",
        "$.id",
        "AS",
        "id",
        "NUMERIC",
        "$.person_id",
        "AS",
        "person_id",
        "NUMERIC",
        "$.embedding",
        "AS",
        "embedding",
        "VECTOR",
        "FLAT",
        "6",
        "DIM",
        str(_EMBEDDING_DIM),
        "DISTANCE_METRIC",
        "COSINE",
        "TYPE",
        "FLOAT32",
    )
    _create_index_if_missing(
        r,
        "idx:logs",
        "log:",
        "SCHEMA",
        "$.id",
        "AS",
        "id",
        "NUMERIC",
        "$.person_id",
        "AS",
        "person_id",
        "NUMERIC",
        "$.recognized_at",
        "AS",
        "recognized_at",
        "TEXT",
        "SORTABLE",
    )
    _create_index_if_missing(
        r,
        "idx:alerts",
        "alert:",
        "SCHEMA",
        "$.id",
        "AS",
        "id",
        "NUMERIC",
        "$.person_id",
        "AS",
        "person_id",
        "NUMERIC",
        "$.is_active",
        "AS",
        "is_active",
        "TAG",
    )


def _create_index_if_missing(r, index_name: str, prefix: str, *args):
    try:
        r.execute_command("FT.INFO", index_name)
    except Exception:
        try:
            r.execute_command(
                "FT.CREATE",
                index_name,
                "ON",
                "JSON",
                "PREFIX",
                "1",
                prefix,
                *args,
            )
            logger.info("Created RediSearch index %s", index_name)
        except Exception as e:
            logger.warning("Failed to create index %s: %s", index_name, e)


# ── Sub-repo implementations ──


class _RedisPersonRepo(PersonRepo):
    def __init__(self, r):
        self._r = r

    def create(self, name: str, **kwargs):
        pid = int(self._r.incr("person_id_seq"))
        now = _now_iso()
        data = {
            "id": pid,
            "name": name,
            "display_name": kwargs.get("display_name"),
            "phone": kwargs.get("phone"),
            "address": kwargs.get("address"),
            "extra_info": kwargs.get("extra_info"),
            "created_at": now,
            "updated_at": now,
        }
        self._r.json().set(f"person:{pid}", "$", data)
        return _to_obj(data)

    def get(self, person_id: int):
        data = self._r.json().get(f"person:{person_id}")
        if data is None:
            return None
        return _to_obj(data)

    def list_all(self):
        keys = sorted(self._r.keys("person:*"))
        result = []
        for key in keys:
            data = self._r.json().get(key)
            if data:
                result.append(_to_obj(data))
        return result

    def update(self, person_id: int, **kwargs):
        key = f"person:{person_id}"
        data = self._r.json().get(key)
        if data is None:
            return None
        for k, v in kwargs.items():
            if k in data:
                data[k] = v
        data["updated_at"] = _now_iso()
        self._r.json().set(key, "$", data)
        return _to_obj(data)

    def delete(self, person_id: int) -> None:
        self._r.delete(f"person:{person_id}")


class _RedisFaceImageRepo(FaceImageRepo):
    def __init__(self, r):
        self._r = r

    def create(
        self,
        person_id: int,
        image_path: str,
        embedding: bytes | None = None,
        capture_condition: str | None = None,
    ):
        fid = int(self._r.incr("face_id_seq"))
        now = _now_iso()
        emb_list = None
        if embedding is not None:
            count = len(embedding) // 4
            emb_list = list(struct.unpack(f"{count}f", embedding))
        data = {
            "id": fid,
            "person_id": person_id,
            "image_path": image_path,
            "capture_condition": capture_condition,
            "embedding": emb_list,
            "created_at": now,
        }
        self._r.json().set(f"face:{fid}", "$", data)
        return _to_obj(data)

    def get_by_person(self, person_id: int):
        keys = self._r.keys("face:*")
        result = []
        for key in keys:
            data = self._r.json().get(key)
            if data and data.get("person_id") == person_id:
                result.append(_to_obj(data))
        return result

    def get_all_embeddings(self):
        keys = self._r.keys("face:*")
        results = []
        for key in keys:
            data = self._r.json().get(key)
            if data and data.get("embedding"):
                emb_floats = data["embedding"]
                emb_bytes = struct.pack(f"{len(emb_floats)}f", *emb_floats)
                results.append({"person_id": data["person_id"], "embedding": emb_bytes})
        return results

    def delete_by_person(self, person_id: int) -> None:
        keys = self._r.keys("face:*")
        for key in keys:
            data = self._r.json().get(key)
            if data and data.get("person_id") == person_id:
                self._r.delete(key)


class _RedisRecognitionLogRepo(RecognitionLogRepo):
    def __init__(self, r):
        self._r = r

    def create(
        self,
        person_id: int | None,
        confidence: float,
        snapshot_path: str | None = None,
    ):
        lid = int(self._r.incr("log_id_seq"))
        now = _now_iso()
        data = {
            "id": lid,
            "person_id": person_id,
            "confidence": confidence,
            "snapshot_path": snapshot_path,
            "recognized_at": now,
        }
        self._r.json().set(f"log:{lid}", "$", data)
        return _to_obj(data)

    def query(
        self,
        start_date=None,
        end_date=None,
        person_id=None,
        page: int = 1,
        page_size: int = 20,
    ):
        keys = sorted(self._r.keys("log:*"), reverse=True)
        results = []
        for key in keys:
            data = self._r.json().get(key)
            if data is None:
                continue
            obj = _to_obj(data)
            if start_date and obj.recognized_at and obj.recognized_at < start_date:
                continue
            if end_date and obj.recognized_at and obj.recognized_at > end_date:
                continue
            if person_id is not None and data.get("person_id") != person_id:
                continue
            results.append(obj)
        offset = (page - 1) * page_size
        return results[offset : offset + page_size]

    def get_stats(self):
        keys = self._r.keys("log:*")
        total = 0
        counts: dict[int | None, int] = {}
        for key in keys:
            data = self._r.json().get(key)
            if data is None:
                continue
            total += 1
            pid = data.get("person_id")
            counts[pid] = counts.get(pid, 0) + 1
        person_stats = [
            {"person_id": pid, "person_name": None, "display_name": None, "count": c}
            for pid, c in counts.items()
        ]
        return {"total_logs": total, "person_stats": person_stats}

    def cleanup(self, retention_days: int) -> int:
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        keys = self._r.keys("log:*")
        deleted = 0
        for key in keys:
            data = self._r.json().get(key)
            if data is None:
                continue
            recognized_at = _parse_iso(data.get("recognized_at"))
            if recognized_at and recognized_at < cutoff:
                self._r.delete(key)
                deleted += 1
        return deleted

    def is_duplicate_log(self, person_id: int, dedup_seconds: int) -> bool:
        cutoff = datetime.utcnow() - timedelta(seconds=dedup_seconds)
        keys = self._r.keys("log:*")
        for key in keys:
            data = self._r.json().get(key)
            if data is None:
                continue
            if data.get("person_id") != person_id:
                continue
            recognized_at = _parse_iso(data.get("recognized_at"))
            if recognized_at and recognized_at >= cutoff:
                return True
        return False


class _RedisAlertRuleRepo(AlertRuleRepo):
    def __init__(self, r):
        self._r = r

    def get_active_rules(self, person_id: int):
        keys = self._r.keys("alert:*")
        results = []
        for key in keys:
            data = self._r.json().get(key)
            if (
                data
                and data.get("person_id") == person_id
                and data.get("is_active") is True
            ):
                results.append(_to_obj(data))
        return results

    def upsert(
        self,
        person_id: int,
        alert_type: str,
        message: str,
        is_active: bool = True,
    ):
        keys = self._r.keys("alert:*")
        for key in keys:
            data = self._r.json().get(key)
            if (
                data
                and data.get("person_id") == person_id
                and data.get("alert_type") == alert_type
            ):
                data["message"] = message
                data["is_active"] = is_active
                self._r.json().set(key, "$", data)
                return _to_obj(data)

        aid = int(self._r.incr("alert_id_seq"))
        data = {
            "id": aid,
            "person_id": person_id,
            "alert_type": alert_type,
            "message": message,
            "is_active": is_active,
        }
        self._r.json().set(f"alert:{aid}", "$", data)
        return _to_obj(data)

    def get_by_person(self, person_id: int):
        keys = self._r.keys("alert:*")
        results = []
        for key in keys:
            data = self._r.json().get(key)
            if data and data.get("person_id") == person_id:
                results.append(_to_obj(data))
        return results

    def delete(self, rule_id: int) -> None:
        self._r.delete(f"alert:{rule_id}")


class _RedisSeqRepo(SeqRepo):
    def __init__(self, r):
        self._r = r

    def next_person_number(self) -> int:
        return int(self._r.incr("person_seq"))


# ── Composite Repository ──


class RedisRepository(AbstractRepository):
    def __init__(self):
        self._r = get_redis_client()
        init_redis_indexes(self._r)
        self._person = _RedisPersonRepo(self._r)
        self._face_image = _RedisFaceImageRepo(self._r)
        self._recognition_log = _RedisRecognitionLogRepo(self._r)
        self._alert_rule = _RedisAlertRuleRepo(self._r)
        self._seq = _RedisSeqRepo(self._r)

    @property
    def person(self) -> _RedisPersonRepo:
        return self._person

    @property
    def face_image(self) -> _RedisFaceImageRepo:
        return self._face_image

    @property
    def recognition_log(self) -> _RedisRecognitionLogRepo:
        return self._recognition_log

    @property
    def alert_rule(self) -> _RedisAlertRuleRepo:
        return self._alert_rule

    @property
    def seq(self) -> _RedisSeqRepo:
        return self._seq
