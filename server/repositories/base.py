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
        embedding_vec: list | None = None,
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


class SeqRepo(ABC):
    @abstractmethod
    def next_person_number(self) -> int: ...


class TrackingEventRepo(ABC):
    @abstractmethod
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
    ): ...

    @abstractmethod
    def get_current_count(self, camera_id: str) -> int: ...

    @abstractmethod
    def get_events(
        self,
        camera_id: str,
        limit: int = 100,
    ) -> list[dict]: ...


class AnimalDetectionLogRepo(ABC):
    @abstractmethod
    def create(
        self,
        class_name: str,
        confidence: float,
        bbox: list[float],
        source: str = "api",
    ): ...

    @abstractmethod
    def query(self, limit: int = 100) -> list: ...


class PlantDetectionLogRepo(ABC):
    @abstractmethod
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
    ): ...

    @abstractmethod
    def query(self, limit: int = 100) -> list: ...


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
    def seq(self) -> SeqRepo: ...

    @property
    @abstractmethod
    def tracking_event(self) -> TrackingEventRepo: ...

    @property
    @abstractmethod
    def animal_detection_log(self) -> AnimalDetectionLogRepo: ...

    @property
    @abstractmethod
    def plant_detection_log(self) -> PlantDetectionLogRepo: ...
