# Workflow — Agent별 업무 분장 및 병렬 실행 계획

> plan.md의 Step 1~19를 5개 Agent에 할당하고, 의존성 기반으로 병렬 실행 가능한 구간을 최대화한 워크플로우.

---

## 1. Agent 구성 요약

| Agent            | 역할                              | 담당 Step              | 핵심 산출물                                                                                                                             |
| ---------------- | --------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **@scaffold**    | 프로젝트 기반 구축                | 1, 2, 3                | 디렉토리, `.env`, `docker-compose.yml`, `requirements.txt`, `.gitignore`                                                                |
| **@backend**     | Repository Pattern + FastAPI 서버 | 4, 5, 6, 8, 9, 10, 11  | `repositories/**`, `config.py`, `database.py`, `models.py`, `schemas.py`, `main.py`, `routers/*`, `alert_service.py` |
| **@face-engine** | DeepFace 핵심 로직                | 7                      | `face_service.py`                                                                                                                       |
| **@frontend**    | Streamlit UI                      | 12, 13, 14, 15, 16, 17 | `ui/app.py`, `ui/pages/*`, `ui/components/*`                                                                                            |
| **@tester**      | 테스트 & 문서화                   | 18, 19                 | `tests/*`, `README.md`                                                                                                                  |

---

## 2. 실행 타임라인 (Gantt 개요)

```
Time ─────────────────────────────────────────────────────────────────────►

Phase A          Phase B                    Phase C              Phase D
(순차)           (부분 병렬)                (완전 병렬)          (병렬)
┌──────────┐    ┌───────────────────────┐  ┌──────────────────┐  ┌──────────┐
│@scaffold │    │@backend               │  │@frontend         │  │@tester   │
│ S1→S2→S3 │───▶│ S4→S5→S6→S8│         │──▶│ S12→S13→S14→    │  │ S18→S19  │
└──────────┘    │            │  ┌──S10──│  │ S15→S16→S17      │  └──────────┘
                │            ├──┤       │  └──────────────────┘       ▲
                │            │  └──S11──│         ▲                   │
                │    ┌───────┘         │         │                   │
                │    │ S9 (recognize) ◄─┤─────────┘                   │
                │    └─────────────────┘                              │
                │                       │                             │
                │  ┌─────────────┐      │                             │
                │  │@face-engine │      │─────────────────────────────┘
                │  │    S7       │──────┘
                │  └─────────────┘
                └───────────────────────┘
```

---

## 3. Phase 상세

### Phase A — 프로젝트 기반 구축 (순차)

> **Agent**: `@scaffold` 단독
> **선행 조건**: 없음
> **후속**: Phase B 전체의 전제 조건

| 순서 | Step    | 태스크                                  | 산출물                                                                                                                                        | 완료 기준                                                           |
| ---- | ------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| A-1  | Step 1  | 디렉토리 구조 + `__init__.py` 전체 생성 | `server/`, `server/routers/`, `server/services/`, `server/repositories/`, `ui/`, `ui/pages/`, `ui/components/`, `face_db/`, `tests/`, `logs/` | 모든 디렉토리 존재, `__init__.py` 포함                              |
| A-2  | Step 1  | `.gitignore` 생성                       | `.gitignore`                                                                                                                                  | `.env`, `face_db/`, `logs/`, `__pycache__/`, `.venv/` 포함          |
| A-3  | Step 2  | `.env` + `.env.example` 생성            | `.env`, `.env.example`                                                                                                                        | `DATABASE_URL` 등 전체 환경변수 포함                           |
| A-4  | Step 3  | `docker-compose.yml` 작성               | `docker-compose.yml`                                                                                                                          | 로장리 Docker 알음 (원격 PostgreSQL 사용)                     |
| A-5  | Step 1  | `requirements.txt` 작성                 | `requirements.txt`                                                                                                                            | 버전 고정, 전체 의존성 목록                                         |
| A-6  | Step 1  | `logger.py` 작성                        | `logger.py`                                                                                                                                   | `setup_logging()` 함수, 콘솔+파일 핸들러                            |
| A-7  | Step 1b | 가상환경 생성 + 패키지 설치             | `.venv/`, 설치된 패키지                                                                                                                       | `python -m venv .venv && pip install -r requirements.txt` 성공      |

**Phase A 검증 체크포인트**:

```bash
python -c "from server.database import engine; engine.connect()"   # PostgreSQL 접속 확인
python -c "from logger import setup_logging; setup_logging()"  # 로거 정상 로드
python --version                       # Python 3.12.x 확인
python -c "import fastapi; print(fastapi.__version__)"         # 패키지 정상 로드
python -c "import deepface; print(deepface.__version__)"       # DeepFace 로드 확인
```

---

### Phase B — 백엔드 + 얼굴 엔진 (부분 병렬)

> **Agent**: `@backend` + `@face-engine` (부분 병렬)
> **선행 조건**: Phase A 완료
> **핵심 원칙**: `@backend`가 DB/ORM/Config 완료 시점에서 `@face-engine`은 병렬 분기

Phase B는 내부적으로 **3개 서브 구간**으로 나뉜다:

#### Phase B-1: Config + Repository + ORM (순차, @backend 단독)

| 순서  | Step          | 태스크                                 | Agent    | 산출물                              | 의존성              |
| ----- | ------------- | -------------------------------------- | -------- | ----------------------------------- | ------------------- |
| B-1.1 | Step 4-A      | Repository 인터페이스 (ABC) 작성       | @backend | `server/repositories/base.py`       | Phase A             |
| B-1.2 | Step 4-B      | Repository 팩토리 함수                 | @backend | `server/repositories/__init__.py`   | B-1.1               |
| B-1.3 | Step 6 (부분) | `server/config.py` — pydantic-settings | @backend | `server/config.py`                  | Phase A (.env 필요) |
| B-1.4 | Step 4-C      | MySQL ORM 모델 정의                    | @backend | `server/models.py`                  | B-1.3               |
| B-1.5 | Step 5        | MySQL DB 엔진/세션 설정                | @backend | `server/database.py`                | B-1.3, B-1.4        |
| B-1.6 | Step 5        | Redis 클라이언트 설정                  | @backend | `server/redis_client.py`            | B-1.3               |
| B-1.7 | Step 5        | MySQL Repository 구현체                | @backend | `server/repositories/mysql_repo.py` | B-1.1, B-1.4, B-1.5 |
| B-1.8 | Step 5        | Redis Repository 구현체                | @backend | `server/repositories/redis_repo.py` | B-1.1, B-1.6        |
| B-1.9 | Step 6 (부분) | Pydantic request/response 스키마       | @backend | `server/schemas.py`                 | B-1.3               |

**Phase B-1 검증 체크포인트**:

```bash
python -c "from server.config import settings; print(settings.DATABASE_URL)"
python -c "from server.repositories import get_repository; repo = get_repository()"
```

#### Phase B-2: 병렬 분기 (@backend 라우터 + @face-engine 동시 진행)

> B-1 완료 후, **@backend**는 라우터/서비스 개발을 계속하고, **@face-engine**은 독립적으로 face_service 개발.

```
                   ┌──── @backend (B-2a) ────────────────────────┐
B-1 완료 ──────────┤                                             ├──▶ B-3
                   └──── @face-engine (B-2b) ────────────────────┘
```

##### B-2a: @backend — 라우터 + 서비스 (face-engine 불필요 부분 먼저)

| 순서   | Step    | 태스크                                                          | 산출물                             | 의존성   |
| ------ | ------- | --------------------------------------------------------------- | ---------------------------------- | -------- |
| B-2a.1 | Step 6  | `server/main.py` — FastAPI 앱 셋업 (lifespan, CORS, 에러핸들러) | `server/main.py`                   | B-1 전체 |
| B-2a.2 | Step 8  | 인물 CRUD 라우터 (`person.py`)                                  | `server/routers/person.py`         | B-2a.1   |
| B-2a.3 | Step 10 | 로그 API 라우터 (`log.py`)                                      | `server/routers/log.py`            | B-2a.1   |
| B-2a.4 | Step 11 | 알림 서비스 (`alert_service.py`)                                | `server/services/alert_service.py` | B-2a.1   |

##### B-2b: @face-engine — DeepFace 핵심 로직 (독립 작업)

| 순서   | Step   | 태스크                                                   | 산출물                            | 의존성                        |
| ------ | ------ | -------------------------------------------------------- | --------------------------------- | ----------------------------- |
| B-2b.1 | Step 7 | `validate_registration_image()` — 등록 품질 검증 (R1~R5) | `server/services/face_service.py` | B-1 (Repository ABC, schemas) |
| B-2b.2 | Step 7 | `validate_detection_frame()` — 검출 품질 필터 (D2~D3)    | 위와 동일 파일                    | B-2b.1                        |
| B-2b.3 | Step 7 | `register_face()` — 검증 → 임베딩 추출 → 이미지 저장     | 위와 동일 파일                    | B-2b.1                        |
| B-2b.4 | Step 7 | `search_face()` — 필터링 → 임베딩 비교 → 다중 얼굴 매칭  | 위와 동일 파일                    | B-2b.2                        |
| B-2b.5 | Step 7 | `check_duplicate()` — 중복 판정 알고리즘 (R6)            | 위와 동일 파일                    | B-2b.3                        |
| B-2b.6 | Step 7 | `register_multi_angle()` — 다중 각도 일괄 등록           | 위와 동일 파일                    | B-2b.5                        |

**@face-engine 내부 의존성**: B-2b.1~B-2b.6은 순차 (같은 파일, 함수 간 의존)

#### Phase B-3: 인식 라우터 연결 (순차, @backend)

> @face-engine 완료 후에만 가능 — `recognition.py` 라우터가 `face_service`를 호출하므로.

| 순서  | Step   | 태스크                                            | Agent    | 산출물                          | 의존성             |
| ----- | ------ | ------------------------------------------------- | -------- | ------------------------------- | ------------------ |
| B-3.1 | Step 9 | 인식/등록 API 라우터 (`recognition.py`)           | @backend | `server/routers/recognition.py` | B-2a.1 + B-2b 전체 |
| B-3.2 | Step 6 | `main.py`에 recognition 라우터 등록 + 모델 워밍업 | @backend | `server/main.py` (수정)         | B-3.1              |

**Phase B 완료 검증 체크포인트**:

```bash
uvicorn server.main:app --reload
# Swagger UI (/docs) 에서:
# - POST /api/persons → 201
# - GET /api/persons → 200
# - POST /api/register (테스트 이미지) → 201 또는 409
# - POST /api/recognize (테스트 이미지) → 200
# - GET /api/logs → 200
```

---

### Phase C — 프런트엔드 + 테스트 (완전 병렬)

> **Agent**: `@frontend` + `@tester` (완전 독립, 동시 진행)
> **선행 조건**: Phase B 완료 (모든 API 엔드포인트 사용 가능)

```
              ┌──── @frontend (C-1) ─────────┐
Phase B 완료 ─┤                              ├──▶ Phase D
              └──── @tester (C-2) ───────────┘
```

#### C-1: @frontend — Streamlit UI

| 순서  | Step    | 태스크                                  | 산출물                                                              | 의존성       |
| ----- | ------- | --------------------------------------- | ------------------------------------------------------------------- | ------------ |
| C-1.1 | Step 12 | 메인 앱 + 사이드바                      | `ui/app.py`, `ui/components/sidebar.py`                             | Phase B 완료 |
| C-1.2 | Step 13 | 실시간 인식 페이지 (`streamlit-webrtc`) | `ui/pages/1_live_recognition.py`, `ui/components/video_renderer.py` | C-1.1        |
| C-1.3 | Step 14 | 얼굴 등록 페이지                        | `ui/pages/2_register_face.py`                                       | C-1.1        |
| C-1.4 | Step 15 | 인물 관리 페이지                        | `ui/pages/3_manage_persons.py`                                      | C-1.1        |
| C-1.5 | Step 16 | 로그 조회 페이지                        | `ui/pages/4_logs.py`                                                | C-1.1        |
| C-1.6 | Step 17 | 설정 페이지 (Config UI)                 | `ui/pages/5_settings.py`                                            | C-1.1        |

> **@frontend 내부 병렬 가능 구간**: C-1.2 ~ C-1.6는 모두 C-1.1에만 의존하므로, **5개 페이지를 동시 개발** 가능.

```
          ┌─ C-1.2 (실시간 인식)
          ├─ C-1.3 (얼굴 등록)
C-1.1 ────┼─ C-1.4 (인물 관리)
          ├─ C-1.5 (로그 조회)
          └─ C-1.6 (설정)
```

#### C-2: @tester — 테스트 케이스 + 문서

| 순서  | Step    | 태스크                                                  | 산출물                          | 의존성       |
| ----- | ------- | ------------------------------------------------------- | ------------------------------- | ------------ |
| C-2.1 | Step 18 | `conftest.py` — pytest fixtures (TestClient, 테스트 DB) | `tests/conftest.py`             | Phase B 완료 |
| C-2.2 | Step 18 | Repository 단위 테스트                                  | `tests/test_repository.py`      | C-2.1        |
| C-2.3 | Step 18 | 인물 API 통합 테스트                                    | `tests/test_person_api.py`      | C-2.1        |
| C-2.4 | Step 18 | 인식 API 통합 테스트                                    | `tests/test_recognition_api.py` | C-2.1        |
| C-2.5 | Step 18 | face_service 단위 테스트 (DeepFace 모킹)                | `tests/test_face_service.py`    | C-2.1        |
| C-2.6 | Step 19 | README.md + `.env.example` 최종 정리                    | `README.md`                     | C-2.1        |

> **@tester 내부 병렬 가능 구간**: C-2.2 ~ C-2.6는 모두 C-2.1에만 의존하므로, **동시 작성** 가능.

```
          ┌─ C-2.2 (Repository 테스트)
          ├─ C-2.3 (Person API 테스트)
C-2.1 ────┼─ C-2.4 (Recognition API 테스트)
          ├─ C-2.5 (Face Service 테스트)
          └─ C-2.6 (README.md)
```

---

### Phase D — 통합 검증

> **Agent**: 모든 Agent 산출물 통합 후 최종 확인
> **선행 조건**: Phase C 완료

| 순서 | 검증 항목          | 실행 명령                              | 기대 결과                                     |
| ---- | ------------------ | -------------------------------------- | --------------------------------------------- |
| D-1  | PostgreSQL 연결 확인  | `python -c "from server.database import engine; engine.connect()"` | PostgreSQL 접속 성공          |
| D-2  | FastAPI 서버 기동  | `uvicorn server.main:app`              | 모델 워밍업 완료, `/docs` 접근 가능           |
| D-3  | 전체 테스트 스위트 | `pytest tests/ -v`                     | 전체 통과                                     |
| D-4  | Streamlit UI 기동  | `streamlit run ui/app.py`              | 5개 페이지 정상 렌더링                        |
| D-5  | 실시간 인식 E2E    | 웹캠 → 인식 → 오버레이                 | 다중 얼굴 각각 이름 + 정보 표시, Unknown 표시 |
| D-6  | 등록 → 중복 체크   | 동일 얼굴 2회 등록                     | 2회차에 409 Conflict                          |
| D-7  | 등록 조건 검증     | 흐릿/작은/다중얼굴 이미지 등록 시도    | 각 조건별 400 + 구체 메시지 반환              |
| D-8  | 알림 규칙          | 규칙 설정 → 인물 감지                  | `st.toast()` 팝업                             |
| D-9  | 환경변수 확인          | `.env` 의 `DATABASE_URL` 점검             | PostgreSQL 접속 정보 일치 확인              |

---

## 4. 병렬 처리 요약

| Phase   | 동시 실행 가능 Agent    | 병렬 태스크 수           | 병목                                   |
| ------- | ----------------------- | ------------------------ | -------------------------------------- |
| **A**   | @scaffold (단독)        | 1                        | 없음 (가벼운 작업)                     |
| **B-1** | @backend (단독)         | 1                        | Config/ORM이 모든 후속의 전제          |
| **B-2** | @backend + @face-engine | **2**                    | @face-engine은 Repository ABC만 의존   || **B-3** | @backend (단독)         | 1                        | recognition 라우터가 face_service 필요 |
| **C**   | @frontend + @tester     | **2** (내부 각 5개 병렬) | 없음 (완전 독립)                       |
| **D**   | 통합 검증               | 1                        | 전체 완료 후                           |

**최대 병렬도**: Phase B-2에서 2 Agent, Phase C에서 2 Agent + 내부 각 5개 태스크 동시 가능.

---

## 5. 의존성 그래프 (전체)

```
[Phase A]
  @scaffold
    ├─ A-1: 디렉토리 구조
    ├─ A-2: .gitignore
    ├─ A-3: .env / .env.example
    ├─ A-4: docker-compose.yml
    ├─ A-5: requirements.txt
    └─ A-6: logger.py
         │
         ▼
[Phase B-1] @backend
    ├─ B-1.1: Repository ABC (base.py)
    ├─ B-1.2: Repository 팩토리 (__init__.py)
    ├─ B-1.3: config.py
    ├─ B-1.4: models.py (ORM)
    ├─ B-1.5: database.py (PostgreSQL 엔진)
    ├─ B-1.7: postgres_repo.py
    └─ B-1.9: schemas.py
         │
         ├────────────────────────┐
         ▼                        ▼
[Phase B-2a] @backend        [Phase B-2b] @face-engine
    ├─ B-2a.1: main.py          ├─ B-2b.1: validate_registration()
    ├─ B-2a.2: person.py        ├─ B-2b.1: validate_registration()
    ├─ B-2a.3: log.py           ├─ B-2b.2: validate_detection()
    └─ B-2a.4: alert_service    ├─ B-2b.3: register_face()
                                 ├─ B-2b.4: search_face()
                                 ├─ B-2b.5: check_duplicate()
                                 └─ B-2b.6: register_multi_angle()
         │                        │
         └────────┬───────────────┘
                  ▼
[Phase B-3] @backend
    ├─ B-3.1: recognition.py
    └─ B-3.2: main.py 라우터 등록 + 워밍업
                  │
         ┌────────┴────────┐
         ▼                  ▼
[Phase C-1] @frontend   [Phase C-2] @tester
    ├─ C-1.1: app.py        ├─ C-2.1: conftest.py
    │   ├─ C-1.2: 실시간     │   ├─ C-2.2: repo 테스트
    │   ├─ C-1.3: 등록       │   ├─ C-2.3: person 테스트
    │   ├─ C-1.4: 관리       │   ├─ C-2.4: recognition 테스트
    │   ├─ C-1.5: 로그       │   ├─ C-2.5: face 테스트
    │   └─ C-1.6: 설정       │   └─ C-2.6: README.md
    └─────────┬──────────────┘
              ▼
[Phase D] 통합 검증 (D-1 ~ D-8)
```

---

## 6. Agent별 체크리스트

### @scaffold 체크리스트

- [ ] 전체 디렉토리 + `__init__.py` 생성
- [ ] `.gitignore` 작성
- [ ] `.env` + `.env.example` (등록/검출 조건 환경변수 포함: `FACE_MIN_CONFIDENCE`, `FACE_MIN_SIZE`, `FACE_BLUR_THRESHOLD`, `FACE_MIN_CONFIDENCE_REALTIME`, `FACE_MIN_SIZE_REALTIME`)
- [ ] `docker-compose.yml` (MySQL/Redis profiles)
- [ ] `requirements.txt` (버전 고정)
- [ ] `logger.py` (setup_logging)
- [ ] **검증**: Docker DB 기동 확인

### @backend 체크리스트

- [ ] `server/config.py` (pydantic-settings)
- [ ] `server/repositories/base.py` (ABC 5종)
- [ ] `server/repositories/__init__.py` (팩토리)
- [ ] `server/models.py` (ORM 5 테이블)
- [ ] `server/database.py` (SQLAlchemy 엔진/세션)
- [ ] `server/redis_client.py` (Redis 연결)
- [ ] `server/repositories/mysql_repo.py`
- [ ] `server/repositories/redis_repo.py`
- [ ] `server/schemas.py` (Pydantic 스키마)
- [ ] `server/main.py` (FastAPI 앱, lifespan, CORS)
- [ ] `server/routers/person.py` (CRUD 5종)
- [ ] `server/routers/log.py` (조회/통계/정리)
- [ ] `server/services/alert_service.py`
- [ ] `server/routers/recognition.py` (인식/등록/다중각도) ← @face-engine 완료 후
- [ ] **검증**: Swagger `/docs` 전체 API 테스트

### @face-engine 체크리스트

- [ ] `validate_registration_image()` — 등록 품질 검증 (R1: 얼굴감지, R2: 신뢰도, R3: 크기, R4: 선명도, R5: 단일얼굴)
- [ ] `validate_detection_frame()` — 검출 품질 필터 (D2: 신뢰도, D3: 크기 필터링)
- [ ] `register_face()` — 검증 파이프라인 → 임베딩 추출 → 이미지 저장
- [ ] `search_face()` — 필터링 → 임베딩 비교 → 다중 얼굴 매칭 (D7)
- [ ] `check_duplicate()` — 중복 판정 (R6: cosine distance + threshold)
- [ ] `register_multi_angle()` — 다중 각도 일괄 등록
- [ ] 등록 실패 시 구체 에러 코드 + 메시지 반환 (R2~R5 각각 다른 메시지)
- [ ] 검출 실패 시 조용히 스킵 (에러 없음)
- [ ] **검증**: 단위 테스트로 등록 조건별 거부/통과 + 검출 필터링 동작 확인

### @frontend 체크리스트

- [ ] `ui/app.py` + `ui/components/sidebar.py` (메인 프레임)
- [ ] `ui/pages/1_live_recognition.py` + `video_renderer.py` — 검출 조건 D1~D7 적용 (streamlit-webrtc, 다중얼굴 오버레이, Unknown 표시)
- [ ] `ui/pages/2_register_face.py` — 등록 조건 R1~R7 에러 표시 (신뢰도/크기/선명도/단일얼굴/중복 각각 st.warning)
- [ ] `ui/pages/3_manage_persons.py` (CRUD + 알림 규칙)
- [ ] `ui/pages/4_logs.py` (필터 + plotly 차트)
- [ ] `ui/pages/5_settings.py` (등록/검출 조건 슬라이더: confidence, min_size, blur_threshold, threshold)
- [ ] **검증**: `streamlit run ui/app.py` → 전 페이지 렌더링 + 등록 거부 메시지 표시 확인

### @tester 체크리스트

- [ ] `tests/conftest.py` (fixtures: TestClient, 테스트 DB)
- [ ] `tests/test_repository.py` (Repository 구현체 단위 테스트)
- [ ] `tests/test_person_api.py` (인물 CRUD 통합 테스트)
- [ ] `tests/test_recognition_api.py` (등록/인식/중복 통합 테스트)
- [ ] `tests/test_face_service.py` — **등록/검출 조건 테스트 포함**:
  - R2: 낮은 신뢰도 이미지 → 거부 확인
  - R3: 작은 얼굴 이미지 → 거부 확인
  - R4: 흐릿한 이미지 → 거부 확인
  - R5: 다중 얼굴 이미지 → 거부 확인
  - D2/D3: 품질 미달 얼굴 필터링 확인
  - D7: 다중 얼굴 각각 독립 매칭 확인
- [ ] `README.md` (설치, 실행, 환경변수 설명 — 등록/검출 조건 환경변수 포함)
- [ ] `.env.example` 최종 검토
- [ ] **검증**: `pytest tests/ -v` → 전체 통과

---

## 7. 크리티컬 패스

> 프로젝트 완료까지 **가장 긴 경로** (병렬화 불가 구간).

```
@scaffold (A-1~A-6)
  → @backend B-1 (config → repo ABC → ORM → DB엔진 → 구현체 → schemas)
    → @face-engine B-2b (face_service.py)
      → @backend B-3 (recognition.py)
        → @frontend C-1 (UI 전체)
          → Phase D (통합 검증)
```

**병목 포인트**:

1. **B-1 (Repository + ORM)**: 모든 후속 작업의 기반. 가장 먼저 완료해야 함.
2. **B-3 (recognition 라우터)**: @face-engine 완료를 대기해야 함. B-2a 라우터들(person, log)을 먼저 완료하여 대기 시간 최소화.
3. **Phase C 진입**: @backend + @face-engine 모두 완료 필요.

---

## 8. 파일-Agent 매핑 (충돌 방지)

> 동일 파일을 2개 이상의 Agent가 수정하지 않도록 소유권 명시.

| 파일                                                            | 소유 Agent       | 비고                      |
| --------------------------------------------------------------- | ---------------- | ------------------------- |
| `.gitignore`, `.env*`, `docker-compose.yml`, `requirements.txt` | @scaffold        | 최초 생성 후 수정 없음    |
| `logger.py`                                                     | @scaffold        |                           |
| `server/config.py`                                              | @backend         |                           |
| `server/database.py`                                            | @backend         |                           |
| `server/redis_client.py` (미사용)                              | @backend         | PostgreSQL 전환으로 미사용 |
| `server/models.py`                                              | @backend         |                           |
| `server/schemas.py`                                             | @backend         |                           |
| `server/repositories/*`                                         | @backend         |                           |
| `server/main.py`                                                | @backend         | B-3에서 라우터 등록 추가  |
| `server/routers/person.py`                                      | @backend         |                           |
| `server/routers/recognition.py`                                 | @backend         | @face-engine 완료 후 작성 |
| `server/routers/log.py`                                         | @backend         |                           |
| `server/services/alert_service.py`                              | @backend         |                           |
| `server/services/face_service.py`                               | **@face-engine** | 유일하게 단독 소유        |
| `ui/**`                                                         | @frontend        | 전체 UI 코드              |
| `tests/**`                                                      | @tester          | 전체 테스트 코드          |
| `README.md`                                                     | @tester          |                           |
