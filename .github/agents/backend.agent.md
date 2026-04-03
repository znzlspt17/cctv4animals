---
description: "Use when: building FastAPI backend, Repository Pattern, database models, ORM, API routers, schemas, config, alert service, recognition router. Backend server development for DeepFace Live."
tools: [read, edit, search, execute, todo]
---

You are @backend — the backend server agent for DeepFace Live. Your job is to build the complete FastAPI server including Repository Pattern abstraction, database layer, API routers, and services.

## Scope

You are responsible for **Phase B-1, B-2a, B-3 (Step 4~6, 8~11)** of the workflow.

### Files You Own

- `server/config.py` — pydantic-settings
- `server/repositories/base.py` — ABC interfaces (PersonRepo, FaceImageRepo, RecognitionLogRepo, AlertRuleRepo, SeqRepo)
- `server/repositories/__init__.py` — `get_repository()` factory
- `server/repositories/mysql_repo.py` — SQLAlchemy PostgreSQL implementation (PostgresRepository)
- `server/models.py` — ORM models (5 tables)
- `server/database.py` — SQLAlchemy engine/session (psycopg2, sslmode=disable)
- `server/redis_client.py` — 미사용 스텁 (수정 불필요)
- `server/schemas.py` — Pydantic request/response schemas
- `server/main.py` — FastAPI app (lifespan, CORS, error handlers, router registration)
- `server/routers/person.py` — Person CRUD (5 endpoints)
- `server/routers/recognition.py` — Recognize/Register/Multi-angle (depends on @face-engine)
- `server/routers/log.py` — Log query/stats/cleanup
- `server/services/alert_service.py` — Alert rule checking

## Constraints

- DO NOT modify `server/services/face_service.py` — owned by @face-engine
- DO NOT modify any `ui/**` files — owned by @frontend
- DO NOT modify any `tests/**` files — owned by @tester
- DO NOT modify `.env`, `docker-compose.yml`, `requirements.txt` — owned by @scaffold
- Service/router layers MUST depend only on `AbstractRepository` — never reference concrete DB code directly
- DO NOT add Redis or MySQL-specific code — PostgreSQL only
- `server/routers/recognition.py` (B-3) can only be built AFTER @face-engine completes `face_service.py`

## Execution Order

### Part 1: Repository Layer (B-1, sequential)

1. **B-1.3** `server/config.py` — pydantic-settings `Settings` class: `DATABASE_URL` and all face recognition/logging env vars
2. **B-1.1** `server/repositories/base.py` — 5 ABC interfaces
3. **B-1.2** `server/repositories/__init__.py` — factory returning `PostgresRepository()`
4. **B-1.4** `server/models.py` — 5 ORM tables (Person, FaceImage, RecognitionLog, AlertRule, PersonNameSeq)
5. **B-1.5** `server/database.py` — SQLAlchemy engine (psycopg2, connect_args sslmode=disable), SessionLocal, get_db, Base
6. **B-1.7** `server/repositories/mysql_repo.py` — PostgresRepository implementing all ABCs
7. **B-1.9** `server/schemas.py` — Pydantic models

### Part 2: API (B-2a, after B-1)

8. **B-2a.1** `server/main.py` — FastAPI app setup (lifespan with DB init + model warmup, CORS, global error handlers, `app.state.repo`)
9. **B-2a.2** `server/routers/person.py` — POST/GET/GET{id}/PUT/DELETE persons
10. **B-2a.3** `server/routers/log.py` — GET logs, GET stats, DELETE cleanup + dedup logic
11. **B-2a.4** `server/services/alert_service.py` — check active rules for person

### Part 3: Recognition Router (B-3, after @face-engine)

14. **B-3.1** `server/routers/recognition.py` — POST /api/recognize, POST /api/register, POST /api/register/multi-angle
15. **B-3.2** Update `server/main.py` — register recognition router + ensure model warmup in lifespan

---

## Key Design Details

### server/config.py — Settings 전체 필드

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # PostgreSQL (원격)
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@100.95.34.69:5555/cctv?sslmode=disable"

    # FastAPI / Streamlit
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    STREAMLIT_PORT: int = 8501

    # DeepFace
    FACE_DB_PATH: str = "face_db"
    DEEPFACE_MODEL: str = "ArcFace"
    DEEPFACE_DETECTOR: str = "retinaface"
    DEEPFACE_DETECTOR_REALTIME: str = "retinaface"
    DEEPFACE_DISTANCE_METRIC: str = "cosine"

    # 등록 조건 (R2~R4)
    FACE_MIN_CONFIDENCE: float = 0.90
    FACE_MIN_SIZE: int = 112
    FACE_BLUR_THRESHOLD: float = 100.0

    # 검출 조건 (D2~D3)
    FACE_MIN_CONFIDENCE_REALTIME: float = 0.80
    FACE_MIN_SIZE_REALTIME: int = 56

    # 공통
    RECOGNITION_THRESHOLD: float = 0.40
    ALLOW_FORCE_REGISTER: bool = False
    RECOGNITION_FRAME_SKIP: int = 3

    # 로깅
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_FILE_ENABLED: bool = True
    LOG_FILE_PATH: str = "logs/app.log"
    LOG_FILE_MAX_BYTES: int = 10485760
    LOG_FILE_BACKUP_COUNT: int = 5
    LOG_RETENTION_DAYS: int = 30
    LOG_DEDUP_SECONDS: int = 10

    class Config:
        env_file = ".env"
```

### Repository ABC — 메서드 시그니처 (base.py)

plan.md Step 4-A의 완전한 시그니처를 그대로 구현:

```python
class PersonRepo(ABC):
    @abstractmethod
    def create(self, name: str, **kwargs) -> Person: ...
    @abstractmethod
    def get(self, person_id: int) -> Person | None: ...
    @abstractmethod
    def list_all(self) -> list[Person]: ...
    @abstractmethod
    def update(self, person_id: int, **kwargs) -> Person: ...
    @abstractmethod
    def delete(self, person_id: int) -> None: ...

class FaceImageRepo(ABC):
    @abstractmethod
    def add(self, person_id: int, image_path: str, embedding: list[float], condition: str) -> FaceImage: ...
    @abstractmethod
    def get_all_embeddings(self) -> list[dict]: ...  # [{person_id, embedding}, ...]
    @abstractmethod
    def delete_by_person(self, person_id: int) -> None: ...

class RecognitionLogRepo(ABC):
    @abstractmethod
    def add(self, person_id: int | None, confidence: float, snapshot_path: str) -> None: ...
    @abstractmethod
    def query(self, filters: dict) -> list: ...  # filters: {start_date, end_date, person_id}
    @abstractmethod
    def cleanup(self, retention_days: int) -> int: ...  # returns deleted count
    @abstractmethod
    def is_duplicate_log(self, person_id: int, dedup_seconds: int) -> bool: ...

class AlertRuleRepo(ABC):
    @abstractmethod
    def get_active_rules(self, person_id: int) -> list: ...
    @abstractmethod
    def upsert(self, person_id: int, alert_type: str, message: str) -> None: ...

class SeqRepo(ABC):
    @abstractmethod
    def next_person_number(self) -> int: ...
```

### Repository Pattern

- PostgreSQL only → `PostgresRepository` (SQLAlchemy ORM, psycopg2-binary)
- Factory: `get_repository()` in `server/repositories/__init__.py` returns `PostgresRepository()`
- File `mysql_repo.py` kept for naming continuity

### PostgreSQL ORM 모델 (models.py) — 관계 및 인덱스

```python
class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=True)
    phone = Column(String(50), nullable=True)
    address = Column(Text, nullable=True)
    extra_info = Column(JSON, nullable=True)  # PostgreSQL: JSONB 자동 매핑
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships (cascade delete)
    face_images = relationship("FaceImage", back_populates="person", cascade="all, delete-orphan")
    recognition_logs = relationship("RecognitionLog", back_populates="person")
    alert_rules = relationship("AlertRule", back_populates="person", cascade="all, delete-orphan")

class FaceImage(Base):
    __tablename__ = "face_images"
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    image_path = Column(String(500), nullable=False)
    capture_condition = Column(String(100), nullable=True)
    embedding = Column(LargeBinary, nullable=True)  # BYTEA: numpy pickle 직렬화, FAISS 캐시용
    created_at = Column(DateTime, default=func.now())

    person = relationship("Person", back_populates="face_images")

class RecognitionLog(Base):
    __tablename__ = "recognition_logs"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True, index=True)
    confidence = Column(Float, nullable=False)
    snapshot_path = Column(String(500), nullable=True)
    recognized_at = Column(DateTime, default=func.now(), index=True)

    person = relationship("Person", back_populates="recognition_logs")

class AlertRule(Base):
    __tablename__ = "alert_rules"
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

    person = relationship("Person", back_populates="alert_rules")

class PersonNameSeq(Base):
    __tablename__ = "person_name_seq"
    id = Column(Integer, primary_key=True, default=1)
    next_val = Column(Integer, nullable=False, default=1)
```

Key ORM 포인트:

- `Person.face_images`: cascade `all, delete-orphan` → Person 삭제 시 FaceImage 자동 삭제
- `RecognitionLog.person_id`: `ondelete="SET NULL"` → Person 삭제 시 로그는 보존 (person_id=NULL)
- `recognized_at`에 인덱스 — 기간 필터 쿼리 최적화
- `person_id` FK 컬럼 전부 인덱스 — JOIN 최적화

### Redis 데이터 구조 (redis_repo.py)

```
# Key 패턴
person:{id}         → RedisJSON (name, display_name, phone, address, extra_info, created_at, updated_at)
face:{id}           → RedisJSON (person_id, image_path, capture_condition, embedding, created_at)
log:{id}            → RedisJSON (person_id, confidence, snapshot_path, recognized_at) + TTL (LOG_RETENTION_DAYS)
alert:{id}          → RedisJSON (person_id, alert_type, message, is_active)
person_seq          → INCR (atomic sequence counter)
face_id_seq         → INCR
log_id_seq          → INCR
alert_id_seq        → INCR

# RediSearch 인덱스
idx:persons         → FT.CREATE ... ON JSON PREFIX 1 person: SCHEMA $.name AS name TAG $.display_name AS display_name TEXT
idx:face_images     → FT.CREATE ... ON JSON PREFIX 1 face: SCHEMA $.person_id AS person_id NUMERIC $.embedding AS embedding VECTOR FLAT 6 DIM 4096 DISTANCE_METRIC COSINE TYPE FLOAT32
idx:logs            → FT.CREATE ... ON JSON PREFIX 1 log: SCHEMA $.person_id AS person_id NUMERIC $.recognized_at AS recognized_at TEXT SORTABLE
idx:alerts          → FT.CREATE ... ON JSON PREFIX 1 alert: SCHEMA $.person_id AS person_id NUMERIC $.is_active AS is_active TAG
```

VSS 벡터 검색 설정:

- DIM: `4096` (VGG-Face embedding dimension)
- DISTANCE_METRIC: `COSINE`
- TYPE: `FLOAT32`
- Algorithm: `FLAT` (brute-force, 소규모 데이터에 적합)

### Pydantic Schemas (schemas.py)

```python
# ── Request 스키마 ──
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
    page_size: int = 50

# ── Response 스키마 ──
class PersonResponse(BaseModel):
    id: int
    name: str
    display_name: str | None
    phone: str | None
    address: str | None
    extra_info: dict | None
    created_at: datetime
    updated_at: datetime

class PersonDetailResponse(PersonResponse):
    face_images: list[FaceImageResponse]

class FaceImageResponse(BaseModel):
    id: int
    person_id: int
    image_path: str
    capture_condition: str | None
    created_at: datetime

class RecognitionResult(BaseModel):
    person_id: int | None
    person_name: str | None
    confidence: float
    distance: float
    bbox: dict  # {x, y, w, h}
    alerts: list[AlertResponse] = []

class RecognizeResponse(BaseModel):
    faces: list[RecognitionResult]

class RegisterResponse(BaseModel):
    person_id: int
    person_name: str
    face_image_id: int
    message: str

class LogResponse(BaseModel):
    id: int
    person_id: int | None
    person_name: str | None
    confidence: float
    snapshot_path: str | None
    recognized_at: datetime

class LogStatsResponse(BaseModel):
    total_count: int
    person_stats: list[dict]  # [{person_id, person_name, count}]

class AlertResponse(BaseModel):
    alert_type: str
    message: str

class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
```

### Sequence 동시성

- MySQL: `SELECT next_val FROM person_name_seq WHERE id=1 FOR UPDATE` → UPDATE → COMMIT
- Redis: `INCR person_seq` (atomic)

### server/main.py — Lifespan 시작 순서

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) 로깅 초기화
    setup_logging()
    logger.info("Starting DeepFace Live server...")

    # 2) DB 초기화 — PostgreSQL
    import server.models  # noqa
    from server.database import Base, engine
    Base.metadata.create_all(bind=engine)  # 테이블 자동 생성
    _ensure_person_seq()                   # person_name_seq 초기 행 보장

    # 3) Repository 팩토리 → app.state
    app.state.repo = get_repository()

    # 4) FaceService (임베딩 캐시 + FAISS 인덱스 빌드)
    from server.services.face_service import face_service
    face_service.warmup()
    face_service.load_embedding_cache(app.state.repo)
    face_service.load_alert_cache(app.state.repo)
    app.state.face_service = face_service

    logger.info("Server ready.")
    yield
    logger.info("Shutting down...")
```

글로벌 에러 핸들러:

```python
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc), "error_code": "FACE_ERROR"})

@app.exception_handler(ConnectionError)
async def db_connection_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": "DB 연결 실패", "error_code": "DB_UNAVAILABLE"})
```

CORS 설정:

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

### API 엔드포인트 상세

#### Person API (`server/routers/person.py`)

| Method | Endpoint            | Request Body   | Response                   | Error |
| ------ | ------------------- | -------------- | -------------------------- | ----- |
| POST   | `/api/persons`      | `PersonCreate` | `PersonResponse` 201       | 500   |
| GET    | `/api/persons`      | —              | `list[PersonResponse]` 200 | —     |
| GET    | `/api/persons/{id}` | —              | `PersonDetailResponse` 200 | 404   |
| PUT    | `/api/persons/{id}` | `PersonUpdate` | `PersonResponse` 200       | 404   |
| DELETE | `/api/persons/{id}` | —              | 204                        | 404   |

- POST: `SeqRepo.next_person_number()` → `person{N}` 이름 생성 → PersonRepo.create()
- DELETE: cascade로 face_images, alert_rules 자동 삭제. face_db 디렉토리 내 이미지 파일도 삭제. 임베딩 캐시 갱신.

#### Recognition API (`server/routers/recognition.py`)

| Method | Endpoint                    | Request                                               | Response                | Error                           |
| ------ | --------------------------- | ----------------------------------------------------- | ----------------------- | ------------------------------- |
| POST   | `/api/recognize`            | `UploadFile`                                          | `RecognizeResponse` 200 | 503 (DB)                        |
| POST   | `/api/register`             | `UploadFile` + `person_id?` + `condition?` + `force?` | `RegisterResponse` 201  | 400 (R1~R5), 409 (R6), 500 (R7) |
| POST   | `/api/register/multi-angle` | `list[UploadFile]` + `person_id?`                     | `RegisterResponse` 201  | 400, 409                        |

등록 에러 코드 매핑:
| 에러 코드 | HTTP | 조건 | 메시지 |
|-----------|------|------|--------|
| `NO_FACE` | 400 | R1 | "얼굴이 감지되지 않았습니다" |
| `LOW_CONFIDENCE` | 400 | R2 | "감지 신뢰도 부족 (재촬영 필요)" |
| `FACE_TOO_SMALL` | 400 | R3 | "얼굴이 너무 작습니다 (가까이 촬영하세요)" |
| `BLURRY_IMAGE` | 400 | R4 | "이미지가 흐릿합니다 (재촬영 필요)" |
| `MULTIPLE_FACES` | 400 | R5 | "한 명만 촬영하세요" |
| `DUPLICATE_FACE` | 409 | R6 | "{matched_name}과(와) {similarity}% 유사합니다" |
| `EMBEDDING_FAIL` | 500 | R7 | "임베딩 추출 실패" |

인식 API 파이프라인 (POST /api/recognize):

1. 이미지 수신 → face_service.search_face() 호출
2. 빈 결과 → `{"faces": []}` 200 (에러 아님)
3. 매칭 결과 있음 → 각 얼굴별 D6 로그 중복 체크 (`repo.is_duplicate_log(person_id, LOG_DEDUP_SECONDS)`)
4. 중복 아닌 경우만 `repo.log.add()` 저장
5. 매칭된 인물의 alert_service 호출 → 활성 알림 포함하여 응답
6. 응답: `RecognizeResponse` with `faces: list[RecognitionResult]`

#### Log API (`server/routers/log.py`)

| Method | Endpoint            | Params / Body                                              | Response                   | Error |
| ------ | ------------------- | ---------------------------------------------------------- | -------------------------- | ----- |
| GET    | `/api/logs`         | `start_date`, `end_date`, `person_id`, `page`, `page_size` | `list[LogResponse]` 200    | —     |
| GET    | `/api/logs/stats`   | `start_date`, `end_date`                                   | `LogStatsResponse` 200     | —     |
| DELETE | `/api/logs/cleanup` | `retention_days?` (default: LOG_RETENTION_DAYS)            | `{"deleted_count": N}` 200 | —     |

- 페이지네이션: `page` (1-based), `page_size` (default 50, max 200)
- D6 중복 억제는 recognition 라우터에서 적용 (로그 저장 시점)

#### Alert Service (`server/services/alert_service.py`)

```python
class AlertService:
    def check_alerts(self, repo: AbstractRepository, person_id: int) -> list[AlertResponse]:
        """인식 시점에 호출. recognition 라우터에서 매칭 결과마다 호출."""
        rules = repo.alert.get_active_rules(person_id)
        return [AlertResponse(alert_type=r.alert_type, message=r.message) for r in rules]
```

호출 시점: `recognition.py`의 POST `/api/recognize`에서 매칭된 각 person_id에 대해 호출. 결과는 `RecognitionResult.alerts`에 포함.

## Verification

```bash
python -c "from server.config import settings; print(settings.DB_BACKEND)"
python -c "from server.repositories import get_repository; repo = get_repository()"
uvicorn server.main:app --reload  # Swagger UI at /docs
```

## Reference Documents

- `plan.md` Phase 2~3 for schema design, API endpoints, Repository Pattern details
- `workflow.md` Phase B for execution order and dependencies
