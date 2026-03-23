from abc import ABC, abstractmethod


class PersonRepo(ABC):
    @abstractmethod
    def create(self, name: str, **kwargs): ...

    @abstractmethod
    def get(self, person_id: int): ...

    @abstractmethod
    def list_all(self): ...

    @abstractmethod
    def update(self, person_id: int, **kwargs): ...

    @abstractmethod
    def delete(self, person_id: int) -> None: ...


class FaceImageRepo(ABC):
    @abstractmethod
    def create(
        self,
        person_id: int,
        image_path: str,
        embedding: bytes | None = None,
        capture_condition: str | None = None,
    ): ...

    @abstractmethod
    def get_by_person(self, person_id: int): ...

    @abstractmethod
    def get_all_embeddings(self): ...

    @abstractmethod
    def delete_by_person(self, person_id: int) -> None: ...


class RecognitionLogRepo(ABC):
    @abstractmethod
    def create(
        self,
        person_id: int | None,
        confidence: float,
        snapshot_path: str | None = None,
    ): ...

    @abstractmethod
    def query(
        self,
        start_date=None,
        end_date=None,
        person_id=None,
        page: int = 1,
        page_size: int = 20,
    ): ...

    @abstractmethod
    def get_stats(self): ...

    @abstractmethod
    def cleanup(self, retention_days: int) -> int: ...

    @abstractmethod
    def is_duplicate_log(self, person_id: int, dedup_seconds: int) -> bool: ...


class AlertRuleRepo(ABC):
    @abstractmethod
    def get_active_rules(self, person_id: int): ...

    @abstractmethod
    def upsert(
        self,
        person_id: int,
        alert_type: str,
        message: str,
        is_active: bool = True,
    ): ...

    @abstractmethod
    def get_by_person(self, person_id: int): ...

    @abstractmethod
    def delete(self, rule_id: int) -> None: ...


class SeqRepo(ABC):
    @abstractmethod
    def next_person_number(self) -> int: ...


class AbstractRepository(ABC):
    """Composite interface — exposes each sub-repo as a property."""

    @property
    @abstractmethod
    def person(self) -> PersonRepo: ...

    @property
    @abstractmethod
    def face_image(self) -> FaceImageRepo: ...

    @property
    @abstractmethod
    def recognition_log(self) -> RecognitionLogRepo: ...

    @property
    @abstractmethod
    def alert_rule(self) -> AlertRuleRepo: ...

    @property
    @abstractmethod
    def seq(self) -> SeqRepo: ...
