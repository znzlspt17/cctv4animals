# Plan: DeepFace Live — 실시간 얼굴 인식 웹 애플리케이션

## TL;DR

웹캠 기반 실시간 얼굴 인식/등록/검색 시스템. FastAPI 백엔드 + Streamlit 프런트엔드 구조로, DeepFace 라이브러리를 활용하여 얼굴을 인식하고 원격 PostgreSQL DB에 메타데이터를 저장한다. **Repository Pattern**을 적용하여 DB 계층을 추상화한다. 이미지 원본은 로컬 파일시스템에 저장하고 DB에는 경로(링크)만 기록. 임베딩 검색은 **FAISS IndexFlatIP** (L2 정규화 후 내적 = cosine similarity) 적용.

---

## Runtime

- **Python**: 3.12.x (3.12.7 권장)
- **가상환경**: venv (`.venv/` 디렉토리, Git 제외)
- **패키지 관리**: pip + `requirements.txt` (버전 고정)
- **OS**: Windows / Linux / macOS

---

## Architecture

```
[Streamlit UI] ←→ [FastAPI Server] ←→ [Repository Layer] ←→ [PostgreSQL 17.x (원격 Docker)]
                        ↕
                   [DeepFace Engine]
                   [FAISS IndexFlatIP]
                        ↕
                [Local Filesystem (face images)]
```

### Repository Pattern 개요

```
server/repositories/
├── base.py           # ABC: PersonRepo, FaceImageRepo, LogRepo, AlertRepo, SeqRepo
├── mysql_repo.py     # SQLAlchemy + PostgreSQL 구현 (PostgresRepository)
└── __init__.py       # get_repository() 팩토리 — PostgresRepository 반환
```

- SQLAlchemy ORM → PostgreSQL (psycopg2-binary, sslmode=disable)
- 서비스/라우터 계층은 `AbstractRepository`만 의존 → 구체 DB 코드를 직접 참조하지 않음

---

## 얼굴 등록/검출 조건 정의

얼굴 이미지를 **저장(등록)**할 때와 실시간 영상에서 **검출(인식)**할 때는 서로 다른 품질 기준이 적용된다. 등록은 고품질 임베딩 확보가 목적이므로 엄격하고, 실시간 검출은 속도와 사용성이 우선이므로 상대적으로 관대하다.

### 1. 등록 조건 (Registration Quality Gate)

얼굴 등록 시 아래 **모든 조건**을 통과해야 DB에 저장된다. 하나라도 미충족 시 사유를 포함한 에러를 반환한다.

| #   | 조건                 | 환경변수                | 기본값     | 설명                                                                                                |
| --- | -------------------- | ----------------------- | ---------- | --------------------------------------------------------------------------------------------------- |
| R1  | **얼굴 감지**        | —                       | —          | 이미지에서 최소 1개의 얼굴이 검출되어야 함 (DeepFace detector)                                      |
| R2  | **감지 신뢰도**      | `FACE_MIN_CONFIDENCE`   | `0.90`     | detector가 반환하는 confidence score 최소 기준. 낮으면 오검출 위험                                  |
| R3  | **최소 얼굴 크기**   | `FACE_MIN_SIZE`         | `112` (px) | 얼굴 bounding box의 가로·세로 최소 픽셀. VGG-Face 입력이 224×224이므로 최소 112px 권장              |
| R4  | **이미지 선명도**    | `FACE_BLUR_THRESHOLD`   | `100.0`    | Laplacian variance 기준. 이 값 미만이면 "흐릿한 이미지" 거부 + 재촬영 안내                          |
| R5  | **단일 얼굴**        | —                       | —          | 등록 이미지에 2개 이상 얼굴 감지 시 거부 ("한 명만 촬영하세요"). UI에서 얼굴 선택 옵션 미제공 (MVP) |
| R6  | **중복 체크**        | `RECOGNITION_THRESHOLD` | `0.40`     | 기존 임베딩 대비 cosine distance가 threshold 이하면 중복 → 409 Conflict                             |
| R7  | **임베딩 추출 성공** | —                       | —          | DeepFace.represent()가 정상적으로 벡터를 반환해야 함                                                |

#### 등록 검증 파이프라인 (순서)

```
[이미지 수신]
  1) 얼굴 감지 (R1) — 미감지 → 400 "얼굴이 감지되지 않았습니다"
  2) 감지 신뢰도 (R2) — confidence < FACE_MIN_CONFIDENCE → 400 "감지 신뢰도 부족 (재촬영 필요)"
  3) 얼굴 크기 (R3) — bbox < FACE_MIN_SIZE → 400 "얼굴이 너무 작습니다 (가까이 촬영하세요)"
  4) 선명도 (R4) — laplacian_var < FACE_BLUR_THRESHOLD → 400 "이미지가 흐릿합니다 (재촬영 필요)"
  5) 단일 얼굴 (R5) — face_count > 1 → 400 "한 명만 촬영하세요"
  6) 임베딩 추출 (R7) — DeepFace.represent() 실패 → 500
  7) 중복 체크 (R6) — 중복 → 409, 통과 → 저장 진행
```

### 2. 검출 조건 (Real-time Recognition Gate)

실시간 인식 시 아래 조건을 적용한다. 등록보다 완화된 기준으로, 미충족 시 **에러 없이 조용히 스킵**한다.

| #   | 조건               | 환경변수                       | 기본값    | 설명                                                             |
| --- | ------------------ | ------------------------------ | --------- | ---------------------------------------------------------------- |
| D1  | **얼굴 감지**      | —                              | —         | 프레임에서 최소 1개 얼굴 검출. 미감지 시 프레임 스킵 (에러 없음) |
| D2  | **감지 신뢰도**    | `FACE_MIN_CONFIDENCE_REALTIME` | `0.80`    | 등록보다 낮은 기준. 빠른 감지 우선                               |
| D3  | **최소 얼굴 크기** | `FACE_MIN_SIZE_REALTIME`       | `56` (px) | 등록의 절반. 먼 거리의 얼굴도 시도                               |
| D4  | **프레임 스킵**    | `RECOGNITION_FRAME_SKIP`       | `3`       | N프레임마다 1회 인식 수행 (CPU 부하 절감)                        |
| D5  | **매칭 임계값**    | `RECOGNITION_THRESHOLD`        | `0.40`    | cosine distance 기준, 이하면 인식 성공                           |
| D6  | **로그 중복 억제** | `LOG_DEDUP_SECONDS`            | `10`      | 동일 인물이 N초 이내 재감되면 로그 스킵                          |
| D7  | **다중 얼굴 처리** | —                              | —         | 프레임 내 여러 얼굴 각각 독립 인식 + 오버레이                    |

#### 실시간 검출 파이프라인 (순서)

```
[프레임 수신]
  1) 프레임 스킵 (D4) — frame_count % RECOGNITION_FRAME_SKIP != 0 → 스킵
  2) 얼굴 감지 (D1) — 미감지 → 스킵 (조용히)
  3) 각 얼굴에 대해:
     a) 감지 신뢰도 (D2) — < FACE_MIN_CONFIDENCE_REALTIME → 해당 얼굴 스킵
     b) 얼굴 크기 (D3) — < FACE_MIN_SIZE_REALTIME → 해당 얼굴 스킵
     c) 임베딩 추출 — 실패 → 해당 얼굴 스킵
  - DB 매칭 (D5) — FAISS `index.search()` → threshold 이내면 인식 성공, 초과면 "Unknown"
     e) 로그 중복 억제 (D6) — 이미 기록된 인물이면 로그 스킵
     f) 알림 체크 — 매칭된 인물의 활성 alert_rule → st.toast()
```

### 3. 등록 vs 검출 조건 비교

| 항목           | 등록 (Registration) | 검출 (Real-time) | 이유                                               |
| -------------- | ------------------- | ---------------- | -------------------------------------------------- |
| 감지 신뢰도    | `0.90` (엄격)       | `0.80` (관대)    | 등록은 고품질 임베딩 필수, 검출은 속도 우선        |
| 최소 얼굴 크기 | `112px`             | `56px`           | 등록은 선명한 얼굴 필요, 검출은 먼 거리도 시도     |
| 이미지 선명도  | 검사함 (Laplacian)  | 검사 안 함       | 실시간은 프레임별 품질 차이가 큼, 검사 시 오버헤드 |
| 다중 얼굴      | 거부 (1명만)        | 모두 처리        | 등록은 명확한 1:1 매핑, 검출은 다중 추적           |
| 실패 처리      | 에러 반환 (400/409) | 조용히 스킵      | 등록은 사용자 피드백 필요, 검출은 UX 유지          |
| 중복 체크      | 수행                | 수행 (로그 억제) | 목적이 다름: 등록은 DB 무결성, 검출은 로그 최적화  |

### 4. 환경변수 요약 (등록/검출 관련)

```env
# ── 등록 조건 ──
FACE_MIN_CONFIDENCE=0.90        # 등록 시 최소 감지 신뢰도
FACE_MIN_SIZE=112               # 등록 시 최소 얼굴 크기 (px)
FACE_BLUR_THRESHOLD=100.0       # 등록 시 선명도 기준 (Laplacian variance)

# ── 검출 조건 ──
FACE_MIN_CONFIDENCE_REALTIME=0.80   # 실시간 인식 최소 감지 신뢰도
FACE_MIN_SIZE_REALTIME=56           # 실시간 인식 최소 얼굴 크기 (px)
RECOGNITION_FRAME_SKIP=3            # N프레임마다 1회 인식

# ── 공통 ──
RECOGNITION_THRESHOLD=0.40          # cosine distance 매칭 임계값
LOG_DEDUP_SECONDS=10                # 동일 인물 로그 억제 시간
ALLOW_FORCE_REGISTER=false          # 중복 강제 등록 허용 여부
```

---

## Phase 1: 프로젝트 기반 구축

### Step 1. 프로젝트 구조 생성

```
deepface_live/
├── .venv/                      # Python 3.12 가상환경 (Git 제외)
├── .env                        # 환경변수 (포트, DB 연결 등)
├── .env.example                # 환경변수 템플릿
├── docker-compose.yml          # 로장리 (원격 PostgreSQL 사용)
├── requirements.txt            # Python 의존성
├── idea.txt                    # (기존)
├── server/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 앱 엔트리포인트
│   ├── config.py               # .env 로딩 (pydantic-settings)
│   ├── database.py             # SQLAlchemy 엔진/세션 설정 (PostgreSQL)
│   ├── redis_client.py         # 스텁 (미사용 — PostgreSQL 전환으로 Redis 제거됨)
│   ├── models.py               # ORM 모델 — PostgreSQL 테이블 (Person, FaceImage, …)
│   ├── schemas.py              # Pydantic request/response 스키마 (공통)
│   ├── repositories/
│   │   ├── __init__.py         # get_repository() 팩토리 함수
│   │   ├── base.py             # ABC: PersonRepo, FaceImageRepo, LogRepo, …
│   │   ├── mysql_repo.py       # PostgreSQL 구현 (PostgresRepository)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── person.py           # 인물 CRUD API
│   │   ├── recognition.py      # 얼굴 인식/등록 API
│   │   └── log.py              # 인식 로그 API
│   └── services/
│       ├── __init__.py
│       ├── face_service.py     # DeepFace 래핑 (등록, 검색, 중복 체크)
│       └── alert_service.py    # 알림 서비스
├── logger.py                       # 로깅 설정 (레벨별 핸들러, 포맷, 파일 출력)
├── ui/
│   ├── app.py                  # Streamlit 메인 앱
│   ├── pages/
│   │   ├── 1_live_recognition.py   # 실시간 인식 페이지
│   │   ├── 2_register_face.py      # 얼굴 등록 페이지
│   │   ├── 3_manage_persons.py     # 인물 관리 페이지
│   │   ├── 4_logs.py              # 인식 로그 조회 페이지
│   │   └── 5_settings.py          # 설정 페이지 (Config UI)
│   └── components/
│       ├── video_renderer.py   # 웹캠 영상 + 얼굴 오버레이 렌더링
│       └── sidebar.py          # 공통 사이드바 컴포넌트
├── .gitignore                  # Git 제외 파일 (.env, face_db/, logs/, __pycache__/ 등)
├── face_db/                    # DeepFace face_db 이미지 디렉토리
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # pytest fixtures (테스트 DB, 클라이언트)
│   ├── test_person_api.py
│   ├── test_recognition_api.py
│   └── test_face_service.py
└── logs/                       # 인식 로그 파일 디렉토리
```

### Step 1b. 가상환경 구성

Python 3.12 기반 venv를 생성하고 의존성을 설치한다 (`uv` 사용 권장).

#### 실행 순서

1. `uv venv .venv --python 3.12` — 가상환경 생성
2. `.venv\Scripts\activate` (Windows) / `source .venv/bin/activate` (Linux/macOS) — 활성화
3. `uv pip install -r requirements.txt` — 전체 의존성 설치

#### 검증

```bash
python --version                                           # Python 3.12.x 확인
pip list | findstr fastapi                                 # fastapi 설치 확인 (Windows)
python -c "import deepface; print(deepface.__version__)"   # DeepFace 로드 확인
```

#### 주의사항

- tensorflow는 deepface의 암묵적 의존성이므로 `requirements.txt`에 명시적으로 핀하여 Python 3.12 호환성을 보장한다.
- numpy 2.x는 tensorflow 2.16과 비호환이므로 `<2.0` 상한을 둔다.
- Python 3.13은 tensorflow 호환이 불확실하므로 3.12.x를 고정한다.

### Step 2. 환경 설정 파일 (.env)

- `DATABASE_URL` (`postgresql+psycopg2://postgres:postgres@100.95.34.69:5555/cctv?sslmode=disable`)
- `FASTAPI_HOST`, `FASTAPI_PORT`
- `STREAMLIT_PORT`
- `FACE_DB_PATH` (얼굴 이미지 저장 디렉토리)
- `DEEPFACE_MODEL` (기본: VGG-Face)
- `DEEPFACE_DETECTOR` (기본: retinaface — 최고 정확도, 측면/어두운 환경 강점)
- `DEEPFACE_DETECTOR_REALTIME` (실시간 인식용 detector, 기본: retinaface. GPU 없으면 ssd 권장)
- `DEEPFACE_DISTANCE_METRIC` (기본: cosine)
- `RECOGNITION_THRESHOLD` (중복 판별 임계값, VGG-Face+cosine 기준 기본: 0.40)
- `ALLOW_FORCE_REGISTER` (중복 판정 무시 강제 등록 허용, 기본: false)
- `RECOGNITION_FRAME_SKIP` (실시간 인식 시 N프레임마다 1회 인식, 기본: 3)
- `FACE_MIN_CONFIDENCE` (등록 시 최소 감지 신뢰도, 기본: 0.90)
- `FACE_MIN_SIZE` (등록 시 최소 얼굴 크기 px, 기본: 112)
- `FACE_BLUR_THRESHOLD` (등록 시 선명도 기준 Laplacian variance, 기본: 100.0)
- `FACE_MIN_CONFIDENCE_REALTIME` (실시간 인식 최소 감지 신뢰도, 기본: 0.80)
- `FACE_MIN_SIZE_REALTIME` (실시간 인식 최소 얼굴 크기 px, 기본: 56)
- `LOG_LEVEL` (앱 전체 로그 레벨, 기본: `INFO`. 옵션: `DEBUG` | `INFO` | `WARNING` | `ERROR` | `CRITICAL`)
- `LOG_FORMAT` (로그 포맷, 기본: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`)
- `LOG_FILE_ENABLED` (파일 로그 활성화, 기본: `true`)
- `LOG_FILE_PATH` (로그 파일 경로, 기본: `logs/app.log`)
- `LOG_FILE_MAX_BYTES` (로그 파일 최대 크기, 기본: `10485760` = 10MB)
- `LOG_FILE_BACKUP_COUNT` (로그 파일 백업 개수, 기본: `5`)
- `LOG_RETENTION_DAYS` (인식 로그 보관 기간, 기본: 30)
- `LOG_DEDUP_SECONDS` (동일 인물 중복 로그 억제 시간, 기본: 10)

#### 로깅 설계 (`logger.py`)

Python 표준 `logging` 모듈 기반. `setup_logging()` 함수를 앱 시작 시 1회 호출하여 전체 로거를 구성한다.

```python
# logger.py
import logging
from logging.handlers import RotatingFileHandler
from server.config import settings

def setup_logging():
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(settings.LOG_FORMAT)

    # 1) 콘솔 핸들러 (항상 활성)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # 2) 파일 핸들러 (LOG_FILE_ENABLED=true 시)
    if settings.LOG_FILE_ENABLED:
        file_handler = RotatingFileHandler(
            settings.LOG_FILE_PATH,
            maxBytes=settings.LOG_FILE_MAX_BYTES,
            backupCount=settings.LOG_FILE_BACKUP_COUNT,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # 3) 외부 라이브러리 노이즈 억제
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
```

**사용 패턴**: 각 모듈에서 `logger = logging.getLogger(__name__)` 생성 후 `logger.info()`, `logger.debug()` 등 사용.

| 레벨       | 용도                  | 예시                                            |
| ---------- | --------------------- | ----------------------------------------------- |
| `DEBUG`    | 개발 중 상세 추적     | 임베딩 벡터 값, 코사인 거리 계산 결과, DB 쿼리  |
| `INFO`     | 정상 동작 기록        | 인물 등록/삭제, 인식 성공, 서버 시작/종료       |
| `WARNING`  | 주의 필요 상황        | 얼굴 미감지 프레임, 임계값 근접 매칭, 캐시 미스 |
| `ERROR`    | 에러 발생 (복구 가능) | DB 연결 재시도, 이미지 저장 실패, API 예외      |
| `CRITICAL` | 치명적 오류           | 모델 로드 실패, DB 완전 단절                    |

### Step 3. Docker Compose

- **원격 PostgreSQL 사용**: `100.95.34.69:5555` (DB: cctv) — 로컬 Docker 컨테이너 불필요
- `docker-compose.yml`은 빈 상태 (`services: {}`) 유지
- 앱(FastAPI + Streamlit)은 로컬 환경에서 직접 실행
- PostgreSQL 테이블은 SQLAlchemy `Base.metadata.create_all()`로 자동 생성 (서버 시작 시)
- `init.sql`: pgvector 확장 활성화 SQL (원격 DB에 `postgresql-17-pgvector` OS 설치 후 실행)

---

## Phase 2: 데이터베이스 설계 및 ORM

### Step 4. 데이터 스키마 설계

#### 4-A. Repository 인터페이스 (`server/repositories/base.py`)

```python
from abc import ABC, abstractmethod

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
    def get_all_embeddings(self) -> list[dict]: ...
    @abstractmethod
    def delete_by_person(self, person_id: int) -> None: ...

class RecognitionLogRepo(ABC):
    @abstractmethod
    def add(self, person_id: int | None, confidence: float, snapshot_path: str) -> None: ...
    @abstractmethod
    def query(self, filters: dict) -> list: ...
    @abstractmethod
    def cleanup(self, retention_days: int) -> int: ...
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

#### 4-B. 팩토리 함수 (`server/repositories/__init__.py`)

```python
from server.repositories.base import AbstractRepository
from server.repositories.mysql_repo import PostgresRepository

def get_repository() -> AbstractRepository:
    return PostgresRepository()
```

라우터/서비스에서는 `get_repository()`로 주입받아 사용. 구체 DB 코드 직접 참조 금지.

#### 4-C. PostgreSQL 테이블 설계

**persons 테이블** (인물 메타데이터)
| Column | Type | Description |
|---|---|---|
| id | SERIAL PK | 고유 ID |
| name | VARCHAR(100) UNIQUE | 자동 생성 이름 (person1, person2...) |
| display_name | VARCHAR(200) | 사용자 지정 표시 이름 (nullable) |
| phone | VARCHAR(50) | 전화번호 (optional) |
| address | TEXT | 주소 (optional) |
| extra_info | JSONB | 추가 정보 (유연한 확장, GIN 인덱스 지원) |
| created_at | TIMESTAMP | 등록일시 |
| updated_at | TIMESTAMP | 수정일시 |

**face_images 테이블** (얼굴 이미지 — 다양한 각도/조명)
| Column | Type | Description |
|---|---|---|
| id | SERIAL PK | 고유 ID |
| person_id | INT FK → persons.id | 소속 인물 |
| image_path | VARCHAR(500) | 로컬 파일 경로 |
| capture_condition | VARCHAR(100) | 촬영 조건 (정면, 좌측, 우측, 밝음, 어두움 등) |
| embedding | BYTEA | 얼굴 임베딩 벡터 (pickle 직렬화, FAISS 캐시용) |
| created_at | TIMESTAMP | 촬영일시 |

**recognition_logs 테이블** (인식 로그)
| Column | Type | Description |
|---|---|---|
| id | BIGSERIAL PK | 고유 ID |
| person_id | INT FK → persons.id (nullable) | 인식된 인물 (미인식 시 NULL) |
| confidence | DOUBLE PRECISION | 유사도 점수 |
| snapshot_path | VARCHAR(500) | 스냅샷 이미지 경로 |
| recognized_at | TIMESTAMP | 인식 시각 |

**alert_rules 테이블** (알림 규칙)
| Column | Type | Description |
|---|---|---|
| id | SERIAL PK | 고유 ID |
| person_id | INT FK → persons.id | 대상 인물 |
| alert_type | VARCHAR(50) | 알림 유형 (toast) |
| message | TEXT | 알림 메시지 |
| is_active | BOOLEAN | 활성 여부 |

**person_name_seq 테이블** (이름 시퀀스)
| Column | Type | Description |
|---|---|---|
| id | INT PK (항상 1) | 단일 행 |
| next_val | INT | 다음 person 번호 |

#### 4-D. 벡터 검색 — FAISS

- **FAISS IndexFlatIP**: L2 정규화 후 내적 연산 = cosine similarity
- 메모리 인 캐시 (`_faiss_index`, `_embedding_dim`): 서버 시작 시 DB 임베딩 로드 후 빌드
- 증분 갱신: 등록 시 `index.add()`, 삭제 시 전체 재빌드 (`_rebuild_faiss_index`)
- `search_face()` → `index.search(vec, k=1)` (단일 최근접 이웃)
- `check_duplicate()` → exclude_person_id 없으면 FAISS, 있으면 numpy fallback
- Docker 배포 시: `faiss-gpu` (CUDA) 사용. Windows 개발환경: `faiss-cpu`

### Step 5. 구현체 작성

- **PostgreSQL 구현** (`server/repositories/mysql_repo.py` → `PostgresRepository`):
  - `server/models.py`에 5개 테이블 ORM 매핑 (SQLAlchemy 2.0)
  - `server/database.py`에 엔진(`psycopg2`, `sslmode=disable`), 세션 팩토리, `Base`
  - `PostgresRepository` 클래스에서 SQLAlchemy Session 사용
  - 파일명은 하위호환 유지를 위해 `mysql_repo.py` 그대로 사용

---

## Phase 3: FastAPI 백엔드 개발

### Step 6. FastAPI 앱 설정 (`server/main.py`)

- **로깅 초기화**: lifespan startup 최초에 `logger.py`의 `setup_logging()` 호출 → 앱 전체 로거 구성
- CORS 미들웨어 (Streamlit origin 허용)
- 라우터 등록
- **lifespan startup**: SQLAlchemy `Base.metadata.create_all()` → 테이블 자동 생성 (PostgreSQL)
- `get_repository()` 팩토리로 생성된 레포지토리를 `app.state.repo`에 저장 → 라우터에서 의존성 주입
- **모델 워밍업**: lifespan startup에서 더미 이미지로 `DeepFace.represent()` 1회 호출하여 RetinaFace + VGG-Face 모델을 미리 메모리에 로딩 (첫 요청 지연 방지)
- **글로벌 에러 핸들러**: 얼굴 미감지 예외(`ValueError`) → 400 응답, DB 연결 실패 → 503 응답

### Step 7. 얼굴 서비스 (`server/services/face_service.py`)

> **이 서비스는 상단 "얼굴 등록/검출 조건 정의" 섹션의 R1~R7 (등록) 및 D1~D7 (검출) 조건을 구현한다.**

- `validate_registration_image(image)` — **등록 품질 검증 (신규 함수)**:
  - R1: 얼굴 감지 여부 확인
  - R2: `detector_confidence >= FACE_MIN_CONFIDENCE` 검증
  - R3: `bbox_width >= FACE_MIN_SIZE and bbox_height >= FACE_MIN_SIZE` 검증
  - R4: Laplacian variance >= `FACE_BLUR_THRESHOLD` (선명도 검증)
  - R5: 감지된 얼굴 수 == 1인지 확인
  - 모든 조건 통과 시 `(True, face_region, confidence)` 반환
  - 미충족 시 `(False, error_code, error_message)` 반환

- `validate_detection_frame(faces)` — **검출 품질 필터 (신규 함수)**:
  - D2: `confidence >= FACE_MIN_CONFIDENCE_REALTIME` 필터링
  - D3: `bbox_size >= FACE_MIN_SIZE_REALTIME` 필터링
  - 통과한 얼굴 목록만 반환 (D7: 다중 얼굴 각각 독립 처리)

- `register_face(image, person_id)`:
  - **`validate_registration_image()` 호출 → 조건 미충족 시 구체 사유 포함 400 에러 반환**
  - DeepFace.represent()로 임베딩 추출 (R7)
  - 기존 DB 임베딩과 비교하여 중복 체크 (R6: threshold 이하면 중복 거부)
  - 이미지를 `face_db/{person_name}/` 디렉토리에 저장
  - face_images 테이블에 레코드 추가

- `search_face(image)` — **실시간 검출용**:
  - DeepFace.represent()로 임베딩 추출
  - **얼굴 미감지 시 빈 결과 반환** (에러 발생하지 않음 — 실시간 인식에서 팝업 폭탄 방지)
  - **`validate_detection_frame()` 으로 품질 미달 얼굴 필터링 후 매칭 수행**
  - **FAISS IndexFlatIP** (`index.search(vec, k=1)`)로 최근접 임베딩 탐색
  - threshold 이내의 가장 유사한 person 반환 (D5)
  - **다중 얼굴: 프레임 내 모든 유효 얼굴 각각 매칭 결과 리스트 반환** (D7)

- `check_duplicate(embedding)`:
  - 기존 등록된 모든 임베딩과의 최소 거리 계산
  - threshold 설정값 이하면 중복으로 판정

#### 중복 등록 방지 상세 설계 (구현 참고)

**1. 판정 기준**

- 새 이미지의 임베딩과 기존 모든 임베딩 간 **cosine distance** 계산
- `RECOGNITION_THRESHOLD` (`.env`) 이하이면 **동일 인물로 판정**
- VGG-Face + cosine 기준 권장 threshold: **0.40** (엄격: 0.35, 관대: 0.50)
- threshold는 Settings UI(Step 17)에서 슬라이더로 조정 가능

**2. 중복 판정 알고리즘 (check_duplicate)**

```python
def check_duplicate(new_embedding: list[float], db: Session) -> tuple[bool, Person | None, float]:
    """
    Returns: (is_duplicate, matched_person, min_distance)
    """
    # 1) 캐시 또는 DB에서 전체 임베딩 로드
    all_faces = get_all_embeddings()  # [{person_id, embedding}, ...]

    if not all_faces:
        return (False, None, float('inf'))

    # 2) numpy 배열로 변환하여 벡터화 연산 (루프 대신 행렬 연산)
    import numpy as np
    new_vec = np.array(new_embedding)
    db_matrix = np.array([f['embedding'] for f in all_faces])

    # 3) cosine distance = 1 - cosine_similarity
    cosine_sim = np.dot(db_matrix, new_vec) / (
        np.linalg.norm(db_matrix, axis=1) * np.linalg.norm(new_vec)
    )
    distances = 1 - cosine_sim

    # 4) 최소 거리 및 해당 person 확인
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    matched_person_id = all_faces[min_idx]['person_id']

    # 5) threshold 비교
    is_duplicate = min_distance <= RECOGNITION_THRESHOLD
    return (is_duplicate, matched_person_id, float(min_distance))
```

**3. 등록 플로우에서의 적용 (register_face)**

```
[이미지 수신]
  → DeepFace.represent() 로 임베딩 추출
  → 얼굴 미감지 시 400 에러 반환
  → check_duplicate(embedding) 호출
    → is_duplicate=True:
        → 409 Conflict 반환
        → 응답에 matched_person 정보 포함 (UI에서 "이미 person3으로 등록됨" 표시)
    → is_duplicate=False:
        → person_id 있으면 해당 인물에 이미지 추가
        → person_id 없으면 새 인물 생성 (시퀀스 증가)
        → face_db/{person_name}/ 에 이미지 파일 저장
        → face_images 테이블 INSERT
        → 임베딩 캐시 갱신
```

**4. 다중 각도 등록 시 (multi-angle)**

- 같은 등록 세션의 여러 이미지는 **서로 간 중복 체크 제외** (자기 자신과 비교 방지)
- 첫 번째 이미지로만 기존 DB 대비 중복 체크 수행
- 통과 시 나머지 이미지는 같은 person_id로 일괄 등록

**5. 경계 사례 처리**
| 상황 | 처리 |
|---|---|
| 마스크/선글라스 착용한 동일 인물 | threshold 범위 밖 → 별도 인물로 등록될 수 있음. 이후 인물 관리에서 수동 병합 |
| 쌍둥이/매우 유사한 타인 | threshold 이하 → 중복으로 오판 가능. UI에서 "유사 인물 발견, 그래도 등록?" 확인 옵션 제공 |
| 저화질/흐릿한 이미지 | 임베딩 품질 저하 → 등록 전 얼굴 감지 confidence 확인, 낮으면 재촬영 안내 |
| DB 비어있음 (첫 등록) | all_faces 빈 리스트 → 중복 체크 스킵, 바로 등록 |

**6. UI 표시 (등록 페이지)**

- 등록 버튼 클릭 전 **"중복 미리보기"** 실행 가능
- 중복 감지 시: 매칭된 인물 이름 + 등록 이미지 썸네일 + 유사도 점수 표시
- `st.warning("person3과 92% 유사합니다. 이미 등록된 인물입니다.")`
- 강제 등록 옵션: `ALLOW_FORCE_REGISTER` (.env, 기본: false) — 관리자가 중복 판정을 무시하고 별도 인물로 등록 가능

**7. 관련 .env 설정**

- `RECOGNITION_THRESHOLD=0.40` — 중복 판정 기준 거리
- `ALLOW_FORCE_REGISTER=false` — 중복 강제 등록 허용 여부

### Step 8. 인물 API (`server/routers/person.py`)

- `POST /api/persons` — 새 인물 등록 (시퀀스로 자동 이름 생성, `SELECT ... FOR UPDATE`로 동시성 안전 보장)
- `GET /api/persons` — 전체 인물 목록
- `GET /api/persons/{id}` — 인물 상세 (얼굴 이미지 목록 포함)
- `PUT /api/persons/{id}` — 인물 정보 수정 (display_name, phone, address, extra_info)
- `DELETE /api/persons/{id}` — 인물 삭제 (연관 이미지, 로그도 정리)

### Step 9. 인식 API (`server/routers/recognition.py`)

- `POST /api/recognize` — 이미지 전송 → 얼굴 인식 결과 반환
- `POST /api/register` — 이미지 + (선택적 person_id) → 얼굴 등록
  - person_id 없으면 새 인물 자동 생성
  - 중복 얼굴이면 409 Conflict 반환
- `POST /api/register/multi-angle` — 여러 각도/조명 이미지를 한 번에 등록

### Step 10. 로그 API (`server/routers/log.py`)

- `GET /api/logs` — 인식 로그 조회 (기간, 인물 필터)
- `GET /api/logs/stats` — 통계 (인물별 인식 빈도 등)
- `DELETE /api/logs/cleanup` — 보관 기간 초과 로그 일괄 삭제 (기본 30일)
- **중복 로그 억제**: 동일 person이 `LOG_DEDUP_SECONDS`(기본 10초) 이내 재감지 시 로그 스킵
- **로그 보관 정책**: `LOG_RETENTION_DAYS` 환경변수로 TTL 설정

### Step 11. 알림 서비스 (`server/services/alert_service.py`)

- 인식 시 해당 person의 alert_rules 확인
- 활성 규칙이 있으면 알림 데이터를 응답에 포함
- Streamlit에서 `st.toast()`로 팝업 표시

---

## Phase 4: Streamlit 프런트엔드 개발

### Step 12. 메인 앱 구조 (`ui/app.py`)

- 멀티페이지 Streamlit 앱 (pages/ 디렉토리 활용)
- 공통 사이드바: 현재 연결 상태, 등록 인물 수 표시

### Step 13. 실시간 인식 페이지 (`ui/pages/1_live_recognition.py`)

> **검출 조건 D1~D7 적용 — 서버 사이드에서 품질 필터링 후 결과만 수신**

- **`streamlit-webrtc`** 패키지로 브라우저 웹캠 → 서버 실시간 프레임 전송 (st.camera_input은 스냅샷 전용이므로 부적합)
- `VideoProcessorBase` 콜백에서 N프레임마다 1회 FastAPI `/api/recognize`에 전송 (D4: RECOGNITION_FRAME_SKIP)
- **API 응답에 다중 얼굴 결과 포함** (D7): 각 얼굴별 bounding box + 인식 결과
- 인식 결과를 영상 위에 오버레이 (이름 + 선택적 정보)
- **Unknown 표시**: 감지되었으나 매칭 실패(D5 미충족)한 얼굴에 "Unknown" + distance 값 표시
- `st.session_state`로 표시 옵션 관리 (이름만 / 이름+전화번호 / 이름+주소 등)
- 알림 규칙에 해당하는 인물 감지 시 `st.toast()` 표시
- **에러 핸들링**: 얼굴 미감지 프레임은 조용히 스킵 (D1: UI 에러 팝업 없음), API 연결 실패 시 상태 배너 표시

### Step 14. 얼굴 등록 페이지 (`ui/pages/2_register_face.py`)

> **등록 조건 R1~R7 적용 — 서버에서 검증 거부 시 구체적 사유를 UI에 표시**

- 웹캠 캡처 또는 이미지 업로드
- **등록 전 사전 검증 피드백** (서버 400 응답 파싱):
  - R2 위반 → `st.warning("감지 신뢰도가 낮습니다. 정면으로 다시 촬영하세요.")`
  - R3 위반 → `st.warning("얼굴이 너무 작습니다. 카메라에 가까이 다가가세요.")`
  - R4 위반 → `st.warning("이미지가 흐릿합니다. 안정적으로 다시 촬영하세요.")`
  - R5 위반 → `st.warning("여러 얼굴이 감지되었습니다. 한 명만 촬영하세요.")`
  - R6 위반 → `st.warning("person3과 92% 유사합니다. 이미 등록된 인물입니다.")` + 강제 등록 옵션
- 다양한 각도 촬영 가이드 (정면, 좌측 45°, 우측 45° 등)
- 등록 전 중복 체크 결과 미리보기
- 추가 정보 입력 폼 (display_name, phone, address 등)

### Step 15. 인물 관리 페이지 (`ui/pages/3_manage_persons.py`)

- 등록된 인물 목록 (테이블/카드 뷰)
- 인물별 등록 이미지 썸네일 보기
- 정보 수정, 삭제 기능
- 알림 규칙 설정

### Step 16. 로그 조회 페이지 (`ui/pages/4_logs.py`)

- 날짜/인물 필터링
- 인식 로그 테이블 + 스냅샷 이미지 표시
- 간단한 통계 차트 (plotly)

### Step 17. 설정 페이지 — Config UI (`ui/pages/5_settings.py`)

- DeepFace 모델/검출기/거리 메트릭 선택
- 인식 threshold 조정 (슬라이더)
- 오버레이 표시 옵션 (이름, 전화번호, 주소 등 토글)
- DB 연결 상태 확인
- PostgreSQL 연결 상태 표시 (`DATABASE_URL` 접속 확인)
- 로그 레벨 설정

---

## Phase 5: 테스트 및 마무리

### Step 18. 테스트 케이스 작성

- `tests/conftest.py`: 테스트용 DB 세션, FastAPI TestClient fixture
- `tests/test_person_api.py`: 인물 CRUD API 테스트
- `tests/test_recognition_api.py`: 등록/인식/중복 체크 API 테스트
- `tests/test_face_service.py`: DeepFace 서비스 단위 테스트 (모킹)
- pytest + httpx AsyncClient 활용

### Step 19. 최종 통합 및 문서화

- README.md 작성 (설치, 실행 방법)
- .env.example 완성

---

## Relevant Files (생성 예정)

| File                                | Purpose                           |
| ----------------------------------- | --------------------------------- |
| `.gitignore`                        | Git 제외 규칙                     |
| `.env` / `.env.example`             | 환경변수 설정                     |
| `docker-compose.yml`                | 빈 파일 (원격 PostgreSQL 사용)    |
| `requirements.txt`                  | Python 의존성                     |
| `server/main.py`                    | FastAPI 엔트리포인트              |
| `logger.py`                         | 로깅 설정 (레벨/포맷/핸들러)      |
| `server/config.py`                  | pydantic-settings 기반 설정 로딩  |
| `server/database.py`                | SQLAlchemy 엔진/세션 (PostgreSQL) |
| `server/redis_client.py`            | 미사용 스텁 (Redis 제거됨)        |
| `server/models.py`                  | ORM 모델 (PostgreSQL)             |
| `server/schemas.py`                 | Pydantic 스키마 (공통)            |
| `server/repositories/__init__.py`   | Repository 팩토리 함수            |
| `server/repositories/base.py`       | 추상 Repository 인터페이스        |
| `server/repositories/mysql_repo.py` | PostgreSQL/SQLAlchemy 구현체      |
| `server/routers/person.py`          | 인물 CRUD 라우터                  |
| `server/routers/recognition.py`     | 얼굴 인식/등록 라우터             |
| `server/routers/log.py`             | 로그 라우터                       |
| `server/services/face_service.py`   | DeepFace 래핑 서비스              |
| `server/services/alert_service.py`  | 알림 서비스                       |
| `ui/app.py`                         | Streamlit 메인                    |
| `ui/pages/1_live_recognition.py`    | 실시간 인식                       |
| `ui/pages/2_register_face.py`       | 얼굴 등록                         |
| `ui/pages/3_manage_persons.py`      | 인물 관리                         |
| `ui/pages/4_logs.py`                | 로그 조회                         |
| `ui/pages/5_settings.py`            | 설정 (Config UI)                  |
| `ui/components/video_renderer.py`   | 영상 오버레이 렌더링              |
| `tests/conftest.py`                 | 테스트 픽스처                     |
| `tests/test_repository.py`          | Repository 구현체 단위 테스트     |
| `tests/test_*.py`                   | 단위/통합 테스트                  |

## Verification

1. 원격 PostgreSQL 연결 확인: `python -c "from server.database import engine; engine.connect(); print('OK')"`
2. `.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload` → Swagger UI(`/docs`) 확인
3. `pytest tests/` 실행 → 전체 테스트 통과 확인
4. `.venv\Scripts\python.exe -m streamlit run ui/app.py` → 웹캠 인식, 등록, 로그 조회 확인
5. 동일 얼굴 중복 등록 시 409 에러 반환 확인
6. 다중 각도 이미지 등록 후 인식 정확도 향상 확인
7. 알림 규칙 설정 후 해당 인물 감지 시 토스트 팝업 확인

## Decisions

- **DB**: 원격 PostgreSQL 17.x (`100.95.34.69:5555`, DB: cctv). Repository Pattern으로 추상화
- **이미지 저장**: 로컬 파일시스템 (`face_db/` 디렉토리), DB에는 경로만 저장
- **알림**: Streamlit `st.toast()` 팝업
- **배포**: PostgreSQL은 원격 Docker, 앱(FastAPI + Streamlit)은 로컬 실행
- **이름 시퀀스**: 별도 테이블 `person_name_seq`로 관리 (`SELECT FOR UPDATE` 동시성 안전)
- **추가 정보**: `extra_info` JSONB 컬럼으로 유연한 확장 지원
- **Face Detector**: RetinaFace (등록/검색 기본). 5-point landmark 기반 정렬로 임베딩 품질 향상. 실시간 인식은 `DEEPFACE_DETECTOR_REALTIME` 설정으로 분리 가능 (GPU 없으면 ssd 권장, 프레임 스킵 전략 병행)
- **실시간 스트리밍**: `streamlit-webrtc` 사용 (브라우저 웹캠 → 서버 프레임 전송). 별도 WebRTC 시그널링 서버 구축은 하지 않음
- **임베딩 검색**: FAISS IndexFlatIP (L2 정규화 + 내적 = cosine similarity). 서버 시작 시 메모리 캐시 빌드. GPU 환경: `faiss-gpu`
- **로그 관리**: `LOG_RETENTION_DAYS` 기반 정리 + 동일 인물 연속 감지 중복 억제 (`LOG_DEDUP_SECONDS`)
- **에러 핸들링**: 등록 시 얼굴 미감지 → 400 에러, 실시간 인식 시 → 조용히 스킵
- **Scope 제외**: 클라우드 배포, 모바일 UI, 외부 WebRTC 시그널링 서버

## Dependencies (requirements.txt) — Python 3.12

> `uv pip install -r requirements.txt` 으로 설치.

```txt
# Core
fastapi>=0.115.0
uvicorn[standard]>=0.32.0

# Database — PostgreSQL
sqlalchemy>=2.0.36
psycopg2-binary>=2.9.9
pgvector>=0.3.0          # pgvector SQLAlchemy 연동 (PostgreSQL 확장 필요)

# Vector Search
faiss-cpu>=1.7.4         # Windows 개발환경용 (Docker 배포시 faiss-gpu 사용)

# Face Recognition
deepface>=0.0.93
tf-keras>=2.16.0         # TensorFlow 2.21+ + RetinaFace 호환 필수
opencv-python>=4.10.0
retina-face>=0.0.17
insightface>=0.7.3
onnxruntime-gpu>=1.17.0
numpy>=1.26.4

# Frontend
streamlit>=1.40.0
streamlit-webrtc>=0.47.0

# Config / HTTP
pydantic-settings>=2.6.0
python-dotenv>=1.0.1
python-multipart>=0.0.9
httpx>=0.28.0

# Utilities
Pillow>=11.0.0
plotly>=5.24.0

# Testing
pytest>=8.3.0
pytest-asyncio>=0.24.0
```

| 패키지           | 비고                                          |
| ---------------- | --------------------------------------------- |
| psycopg2-binary  | PostgreSQL driver, `sslmode=disable` 필요     |
| pgvector         | pgvector 확장 SQLAlchemy 연동 (선택적)        |
| faiss-cpu        | 개발환경. Docker 배포 시 `faiss-gpu` 교체     |
| tf-keras         | TF 2.21+에서 RetinaFace 필수 의존             |
| deepface         | ArcFace 모델 사용 (512-dim embedding)         |
| onnxruntime-gpu  | InsightFace GPU 추론                          |
| 나머지           | 최신 안정                  | 3.12 문제 없음                        |

---

## Implementation Workflow — Agent 업무 분장

### Agent 구성 (5개)

| Agent            | 역할                              | 담당 Step      | 핵심 산출물                                                                                                                                                                                                        |
| ---------------- | --------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **@scaffold**    | 프로젝트 기반 구축                | Step 1~3       | `.gitignore`, `.env`, `.env.example`, `docker-compose.yml`, `requirements.txt`, 디렉토리/`__init__.py`                                                                                                             |
| **@backend**     | Repository Pattern + FastAPI 서버 | Step 4~6, 8~11 | `server/repositories/**`, `server/config.py`, `server/database.py`, `server/redis_client.py`, `server/models.py`, `server/schemas.py`, `server/main.py`, `server/routers/*.py`, `server/services/alert_service.py` |
| **@face-engine** | DeepFace 핵심 로직                | Step 7         | `server/services/face_service.py`                                                                                                                                                                                  |
| **@frontend**    | Streamlit UI 전체                 | Step 12~17     | `ui/app.py`, `ui/pages/*.py`, `ui/components/*.py`                                                                                                                                                                 |
| **@tester**      | 테스트 및 문서화                  | Step 18~19     | `tests/*.py`, `README.md`                                                                                                                                                                                          |

### 실행 순서 및 의존성

```
Phase A (순차):
  @scaffold ──→ @backend (DB/ORM 먼저 → schemas → main → routers → alert)

Phase B (부분 병렬):
  @backend DB/ORM 완료 시점에서:
  ┌─ @face-engine ──────┐
  │                     │
  └─ @backend 라우터 ───┘  ← recognition 라우터만 face-engine 완료 후 연결

Phase C (병렬):
  @backend + @face-engine 완료 후:
  ┌─ @frontend
  └─ @tester              ← 동시 진행 가능
```

### Agent별 상세 태스크

#### @scaffold — 프로젝트 기반 구축

**선행**: 없음

1. 디렉토리 구조 전체 생성 (`server/`, `server/routers/`, `server/services/`, `server/repositories/`, `ui/`, `ui/pages/`, `ui/components/`, `face_db/`, `tests/`, `logs/`)
2. 모든 `__init__.py` 생성
3. `.gitignore` (`.env`, `face_db/`, `logs/`, `__pycache__/`, `.venv/`)
4. `.env.example` + `.env` (plan Step 2의 모든 환경변수 — `DB_BACKEND`, MySQL/Redis 설정 포함)
5. `docker-compose.yml` (MySQL + Redis, profiles로 분리)
6. `requirements.txt` (버전 고정)
7. 가상환경 생성 + 패키지 설치 (`python -m venv .venv && pip install -r requirements.txt`)

**검증**: `docker compose --profile mysql up -d` → MySQL 기동 성공, `python -c "import fastapi"` → 패키지 정상 로드

---

#### @backend — Repository Pattern + FastAPI 서버

**선행**: @scaffold

**Part 1: Repository Layer (Step 4~5)**

1. `server/config.py` — pydantic-settings `Settings` 클래스 (`.env` 전체 매핑, `DB_BACKEND` 포함)
2. `server/repositories/base.py` — ABC 5개 (`PersonRepo`, `FaceImageRepo`, `RecognitionLogRepo`, `AlertRuleRepo`, `SeqRepo`)
3. `server/repositories/__init__.py` — `get_repository()` 팩토리 함수 (`DB_BACKEND` 분기)
4. **MySQL 구현체**:
   - `server/models.py` — 5개 테이블 ORM (`Person`, `FaceImage`, `RecognitionLog`, `AlertRule`, `PersonNameSeq`)
   - `server/database.py` — SQLAlchemy 엔진, `SessionLocal`, `get_db`, `Base`
   - `server/repositories/mysql_repo.py` — `MySQLRepository` (ABC 구현)
   - 관계: `Person.face_images` (1:N), `cascade="all, delete-orphan"`
   - 인덱스: `face_images.person_id`, `recognition_logs.(person_id, recognized_at)`
5. **Redis 구현체**:
   - `server/redis_client.py` — Redis 연결 관리
   - `server/repositories/redis_repo.py` — `RedisRepository` (ABC 구현)
   - JSON key 스키마 정의 + RediSearch 인덱스 생성 (`FT.CREATE`)
   - 벡터 검색은 VSS FLAT/cosine 사용

**Part 2: API (Step 6, 8~11)**

6. `server/schemas.py` — Pydantic 모델 (공통 — DB 백엔드 무관)
7. `server/main.py` — CORS, 라우터 마운트, lifespan (`DB_BACKEND` 분기 초기화 + warmup + cache reload), `app.state.repo` 주입, 글로벌 에러 핸들러
8. `server/routers/person.py` — CRUD 5개 엔드포인트 (레포지토리 사용)
9. `server/routers/recognition.py` — recognize, register, register/multi-angle (face-engine 완료 후 연결)
10. `server/routers/log.py` — 조회, 통계, cleanup + 중복억제 로직
11. `server/services/alert_service.py` — 인물별 활성 알림 규칙 조회

**검증**: `DB_BACKEND=mysql` 및 `DB_BACKEND=redis` 각각으로 Swagger UI(`/docs`)에서 전 API 호출 성공

---

#### @face-engine — DeepFace 핵심 로직

**선행**: @backend Part 1 완료 (repositories/base.py, 구현체 참조)

1. `FaceService` 클래스 (싱글톤):
   - Repository 인터페이스를 주입받아 사용 (구체 DB 코드 직접 참조 금지)
   - `_embedding_cache: dict` — 앱 시작 시 레포지토리 통해 전체 로드
   - `warmup()` — 더미 이미지로 모델 프리로드
   - `extract_embedding(image_bytes)` — DeepFace.represent() 래핑
   - `check_duplicate(embedding)` — 레포지토리를 통해 임베딩 조회 + cosine distance (plan 알고리즘)
   - `register_face(image_bytes, person_id?, condition?)` — 중복체크 → 저장 → 캐시 갱신
   - `register_multi_angle(images, person_id?)` — 첫 이미지만 중복체크, 나머지 일괄
   - `search_face(image_bytes)` — 캐시 대비 검색 (미감지 → 빈 dict)
   - `get_next_person_name()` — 레포지토리의 `SeqRepo.next_person_number()` 사용
   - `reload_cache()`, `delete_person_faces(person_id)`

**검증**: 이미지 등록/검색/중복 체크 스크립트 동작

---

#### @frontend — Streamlit UI

**선행**: @backend + @face-engine 완료

1. `ui/components/sidebar.py` — 연결 상태, 등록 인원 수
2. `ui/components/video_renderer.py` — 바운딩 박스 + 텍스트 오버레이
3. `ui/app.py` — 멀티페이지 설정
4. `ui/pages/1_live_recognition.py` — streamlit-webrtc, VideoProcessor, frame_skip, toast 알림
5. `ui/pages/2_register_face.py` — 촬영/업로드, 다중 각도 가이드, 중복 미리보기, 등록 폼
6. `ui/pages/3_manage_persons.py` — 인물 목록/수정/삭제, 알림 규칙 관리
7. `ui/pages/4_logs.py` — 필터링, 테이블, plotly 통계 차트
8. `ui/pages/5_settings.py` — threshold 슬라이더, 오버레이 토글, DB 연결 테스트

**검증**: `streamlit run ui/app.py` → 5개 페이지 렌더링 + API 연동

---

#### @tester — 테스트 및 문서화

**선행**: @backend + @face-engine 완료 (frontend와 병렬 가능)

1. `tests/conftest.py` — 테스트용 Repository Mock/Stub, TestClient, 더미 이미지, face_service mock
2. `tests/test_person_api.py` — CRUD + cascade 삭제 (MySQL 및 Redis 모드 각각 테스트)
3. `tests/test_recognition_api.py` — 등록, 중복(409), 미감지(400), 인식, 다중 각도
4. `tests/test_face_service.py` — extract_embedding, check_duplicate 경계값, 캐시
5. `tests/test_repository.py` — Repository 구현체 단위 테스트 (MySQL repo + Redis repo)
6. `README.md` — 설치/실행/API/환경변수/DB_BACKEND 교체 설명

**검증**: `pytest tests/ -v` → 전체 통과

---

### 검증 체크포인트

| 시점                       | 검증 항목                                                            |
| -------------------------- | -------------------------------------------------------------------- |
| Phase A: @scaffold 완료    | `docker compose --profile mysql up -d` 성공, Python import 에러 없음 |
| Phase B: @backend Part 1   | 두 백엔드 모두 초기화 동작 (MySQL: 테이블 생성, Redis: 인덱스 생성)  |
| Phase B: @face-engine 완료 | 더미 이미지 등록/검색 동작                                           |
| Phase B: @backend 완료     | `DB_BACKEND=mysql` 및 `redis` 모두 Swagger UI 전 API 호출            |
| Phase C: @tester 완료      | `pytest tests/ -v` 전체 통과                                         |
| Phase C: @frontend 완료    | Streamlit 5개 페이지 + API 연동                                      |
| **전체 완료**              | Verification 7개 항목 모두 통과                                      |
