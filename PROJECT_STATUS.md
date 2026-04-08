# DeepFace Live — 프로젝트 현황 보고서

> 최종 업데이트: 2026-04-09  
> 기준 브랜치: `4animals` (로컬)

---

## 1. 서비스 구성 개요

| 서비스 | 실행 방법 | 기본 포트 | 역할 |
|--------|-----------|-----------|------|
| FastAPI | `uv run uvicorn server.main:app --host 0.0.0.0 --port 8000` | 8000 | REST API + 백그라운드 카운팅 |
| Streamlit | `uv run streamlit run app.py --server.port 8501` | 8501 | 테스트 UI (얼굴 등록/인식, 모델 탐지) |

### 서버 기동 순서 (lifespan)

```
1. 로깅 초기화
2. PostgreSQL 테이블 생성 (Base.metadata.create_all)
3. Repository 팩토리 → app.state.repo
4. FaceService warmup + 임베딩 캐시 로드 (FAISS 인덱스 빌드)
5. CameraManager.start_all() → CAMERAS_JSON 기반 카메라 스레드 시작
6. AnimalService warmup (YOLO)
7. PlantService warmup (FasterRCNN)
8. LettuceService warmup (콘보루션널 분류모델)
```

---

## 2. 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────┐
│  Streamlit (app.py :8501)                        │
│  · 얼굴 등록/인식  · 동물/식물 탐지  · 라인 설정  │
│  pages/animal_crossing_webcam.py (WebRTC 실시간) │
└──────────────────┬───────────────────────────────┘
                   │ HTTP (FASTAPI_PUBLIC_HOST 기반 URL)
┌──────────────────▼───────────────────────────────┐
│  FastAPI (server/main.py :8000)                  │
│                                                  │
│  /api/recognize   /api/register   /api/persons   │
│  /api/count/*     /api/animal/*   /api/plant/*   │
│  /api/lettuce/*   /api/logs/*                    │
│                                                  │
│  ┌──────────────┐  ┌────────────────────────┐   │
│  │  FaceService │  │    CameraManager       │   │
│  │  ArcFace 512 │  │  N대 카메라 관리        │   │
│  │  FAISS InMem │  │  → PersonCounterService│   │
│  └──────────────┘  └────────────────────────┘   │
│  ┌──────────────┐  ┌───────────────────────┐    │
│  │ AnimalService│  │     PlantService       │   │
│  │ YOLO (m-hpo) │  │  FasterRCNN ConvNeXt  │   │
│  └──────────────┘  └───────────────────────┘    │
│                 Repository (PostgreSQL)           │
└──────────────────────────────────────────────────┘
                   │ publish_event()
       ┌───────────▼──────────┐
       │  외부 이벤트 서버     │
       │ 172.16.15.43:8000    │
       └──────────────────────┘
```

---

## 3. 서버 역할 및 입력 모델

### 이 서버의 실제 역할

이 서버는 **AI 엣지 서버**로, 다음 세 가지에 집중합니다:

1. **AI 추론**: 영상 프레임에서 사람·얼굴·동물·식물 분석
2. **결과 저장**: PostgreSQL (pgvector 포함) 에 추론 결과 적재
3. **이벤트 전송**: 메인 서버(`172.16.15.43:8000`)에 탐지 이벤트 푸시

### 카메라 입력 모델 — 데모 vs 운영

| 항목 | 현재 (데모) | 실제 운영 |
|------|-------------|-----------|
| 카메라 접근 | `cv2.VideoCapture(0)` 직접 열기 | OS/전용 SW가 RTSP 스트림 제공 |
| 프레임 공급 | `PersonCounterService` 내부 루프 | 외부 소프트웨어 → RTSP or HTTP POST |
| 얼굴·동물·식물 | `POST /api/recognize` 등으로 외부에서 프레임 전송 | 동일 방식 유지 |
| 제어 권한 | 이 서버가 카메라 직접 점유 | OS 레이어에서 카메라 독점 관리 |

---

## 4. 멀티카메라 확장 (2026-04-07 적용)

### 변경 내용

| 파일 | 변경 |
|------|------|
| `server/services/person/camera_config.py` | **신규** — 카메라 1대 설정 dataclass |
| `server/services/person/camera_manager.py` | **신규** — N대 PersonCounterService 통합 관리 |
| `server/services/person/person_service.py` | 전역 싱글톤 제거, `CameraConfig`를 생성자에서 수신 |
| `server/services/person/__init__.py` | `CameraConfig`, `CameraManager` 익스포트로 교체 |
| `server/config.py` | `CAMERAS_JSON` 환경변수 추가, `get_camera_configs()` 헬퍼 |
| `server/main.py` | `CameraManager.start_all()` / `stop_all()` 사용 |
| `server/routers/person_count.py` | 멀티카메라 API 전면 재작성 + 하위호환 유지 |
| `server/schemas.py` | `CameraConfigRequest`, `CameraStatusResponse`, `CameraAggregateResponse` 추가 |

### 카메라 설정 방법

**.env (정적, 서버 시작 시 자동 로드)**
```env
# 단일 카메라 (기본 폴백)
PERSON_VIDEO_SOURCE=0
PERSON_CAMERA_ID=cam_01

# 멀티카메라 (CAMERAS_JSON 우선 적용)
CAMERAS_JSON=[
  {"camera_id":"cam_01","video_source":"0","label":"정문"},
  {"camera_id":"cam_02","video_source":"rtsp://192.168.1.10/stream","label":"후문","line_start_y":540,"line_end_y":540}
]
```

**런타임 동적 추가**
```http
POST /api/count/cameras
Content-Type: application/json
{"camera_id":"cam_03","video_source":"1","label":"측문"}
```

### API 목록

```
GET    /api/count/cameras                          전체 카메라 상태 목록
POST   /api/count/cameras                          런타임 카메라 추가
DELETE /api/count/cameras/{camera_id}              카메라 제거
GET    /api/count/cameras/{camera_id}/status       특정 카메라 상태
GET    /api/count/cameras/{camera_id}/events       특정 카메라 이벤트
POST   /api/count/cameras/{camera_id}/reset        카운트 초기화
POST   /api/count/cameras/{camera_id}/pause        카메라 일시 해제
POST   /api/count/cameras/{camera_id}/resume       카메라 재개
GET    /api/count/aggregate                        전체 합산 인원수
GET    /api/count/status                           첫 번째 카메라 상태 (하위 호환)
POST   /api/count/camera/pause                     첫 번째 카메라 해제 (하위 호환)
POST   /api/count/camera/resume                    첫 번째 카메라 재개 (하위 호환)
```

---

## 5. 변경 이력

### 현재 진행 중 (미코및)

| 파일 | 내용 |
|------|------|
| `server/services/plant/lettuce_service.py` | **신규** — 콘보루션널 기반 상추 병해 탐지 서비스 (`best_model_lettuce.pt`) |
| `server/routers/lettuce.py` | **신규** — `/api/lettuce/detect`, `/api/lettuce/status` 엔드포인트 |
| `server/main.py` | LettuceService warmup 돵립 실행 (단계 8) + `lettuce.router` 등록 |
| `server/config.py` | `FASTAPI_PUBLIC_HOST` 기본값 `""` → `"172.16.30.124:8000"` 변경 |
| `lettuce_checkpoint_loader.py` | 상추 모델 체크포인트 로더 유틸리티 |
| `lettuce_infer.ipynb` | 상추 탐지 노트북 (개발용) |

> **동작**: `POST /api/lettuce/detect` → 콨볳 영역 크롭 → `plant_detection_logs` 저장 (crop_type=11/상추) → 외부 서버 전송  
> 병해가 있으면 정상(normal) 탐지는 제외하고 병해만 bc0틸.

### 2026-04-09 — localhost 하드코딩 제거

| 파일 | 변경 내용 |
|------|----------|
| `app.py` | `_API_BASE` 변수를 `settings.FASTAPI_PUBLIC_HOST` 기반으로 생성. 미설정 시 `http://localhost:8000` 폴백 (탭3·탭4·탭6 동물/식물 API 호출에 적용) |
| `pages/animal_crossing_webcam.py` | `_API_BASE = "http://localhost:8000"` 하드코딩 제거 → 동일 패턴 적용 |
| `server/routers/animal.py` | `FASTAPI_PUBLIC_HOST` 미설정 시 `_base_url = None`. 이미지 URL을 `None`으로 두어 외부 서버에 localhost URL이 전송되지 않도록 수정 |
| `server/routers/plant.py` | 동일 — `FASTAPI_PUBLIC_HOST` 미설정 시 `_base_url = None`, `_img_url = None` |

> **배경**: 이전에는 `FASTAPI_PUBLIC_HOST` 미설정 시 `request.base_url`(≒ `http://localhost:8000/`)을 폴백으로 사용하여 외부 이벤트 서버에 접근 불가한 URL이 전송되는 버그가 있었음. 폴백을 `None`으로 변경하여 외부 전송 시 이미지 URL 필드 자체를 생략하도록 수정.

### 2026-04-07 — 프로젝트 구조 정리

| 항목 | 이유 | 조치 |
|------|------|------|
| `tests/test_arcface_distinction.py` | pytest 대상 아님 (진단 스크립트) | `scripts/diag/diag_arcface_distinction.py` 로 이동 |
| `tests/test_arcface_verify.py` | pytest 대상 아님 (진단 스크립트) | `scripts/diag/diag_arcface_verify.py` 로 이동 |
| `tests/test_embedding_diag.py` | Buffalo_L(구버전) 사용, 진단 스크립트 | `scripts/diag/diag_embedding_diag.py` 로 이동 |
| `tests/test_embedding_diag2.py` | Buffalo_L(구버전) 사용, 진단 스크립트 | `scripts/diag/diag_embedding_diag2.py` 로 이동 |
| `tests/test_model_compare.py` | Buffalo_L(구버전) 사용, 진단 스크립트 | `scripts/diag/diag_model_compare.py` 로 이동 |
| `schemas.AlertResponse` | 미사용 (참조 코드 없음) | `server/schemas.py` 에서 제거 |
| `README.md` 구 인원카운팅 API | `GET /count/current` 미존재 엔드포인트 기재 | README 전면 재작성 |
| `pyproject.toml` description | 기본값 "Add your description here" | 프로젝트 설명으로 교체 |

---

## 6. 현재 이슈 / 알려진 제한사항

### ✅ 해결됨

| 항목 | 내용 | 해결 |
|------|------|------|
| `person_counter_service` import 오류 | `__init__.py`가 삭제된 싱글톤을 임포트 | `CameraManager`/`CameraConfig`로 교체 |
| `recognition.py` 타입 오류 | `person_id: int \| None`을 `RegisterResponse` 에 직접 전달 | `assert person_id is not None` 추가 |
| 진단 스크립트가 pytest 오염 | `test_*.py` 형식으로 `tests/` 에 배치 | `scripts/diag/` 로 이동 완료 |
| 동물/식물 이미지 URL에 localhost 전송 | `FASTAPI_PUBLIC_HOST` 미설정 시 `request.base_url` 폴백으로 외부 서버에 localhost URL 전송 | `_base_url = None` 처리로 이미지 URL 필드 생략 |

### ⚠️ 주의 / 잠재 이슈

| 항목 | 수준 | 내용 |
|------|------|------|
| PersonDetector ROI 하드코딩 | **중** | `detector.py`가 `settings.PERSON_ROI_*` 전역값 사용. `CameraConfig.roi_*` 필드가 실제 `PersonDetector`에 미전달 |
| 외부 이벤트 서버 미응답 | **낮음** | `RESULT_PUBLISHER_BASE_URL` 미응답 시 로그 경고만 (비차단) |
| DB 세션 범위 | **중** | `get_events()`가 세션 밖에서 ORM 객체 반환 — 관계 필드 접근 시 `DetachedInstanceError` 가능. 현재는 단순 컬럼만 접근하므로 동작 |
| Streamlit WebRTC 카메라 경합 | **중** | `cam_01`이 USB 카메라 상시 점유. WebRTC 페이지 진입 전 자동으로 `POST /api/count/camera/pause` 호출하나, 타임아웃 가능 |
| `face_image_id` 캐시 누락 | **낮음** | `get_all_embeddings()`가 `face_image_id` 미반환 → 캐시 항목 `face_image_id=None` |
| `FASTAPI_PUBLIC_HOST` 미설정 시 이미지 URL 없음 | **낮음** | 설정 미입력 시 동물/식물 탐지 외부 전송에 `detect_image_url` 포함 안 됨. 기본값(`172.16.30.124:8000`)이 실제 환경과 다를 수 있음 |

### ❌ 미구현

| 항목 | 내용 |
|------|------|
| 카메라별 PersonDetector ROI 적용 | `CameraConfig.roi_*` → `PersonDetector` 전달 로직 |
| 스냅샷 저장 | `snapshot_path = ""` 하드코딩 — 탐지 이벤트 이미지 미저장 |
| 인식 로그 중복 제거 활성화 | `is_duplicate_log()` 구현됨, 라우터에서 미호출 |
| 동물 WebRTC DB 저장 | WebRTC 페이지 탐지 이벤트 후 DB 저장 미구현 |

---

## 7. 데이터베이스 스키마 요약

```
persons               face_images              recognition_logs
─────────────────     ────────────────────     ─────────────────────
id (PK)               id (PK)                  id (BigInt PK)
name (unique)         person_id (FK→persons)   person_id (FK, nullable)
display_name          image_path               confidence
phone                 embedding (BYTEA)        snapshot_path
address               embedding_vec (vector)   recognized_at
extra_info (JSON)     capture_condition
created_at            created_at

tracking_events              animal_detection_logs    plant_detection_logs
────────────────────         ─────────────────────    ────────────────────
id (BigInt PK)               id / source              id / source
camera_id                    class_name               class_name / disease_*
tracker_id / direction       confidence               confidence
count_change / current_count bbox_* / detected_at     bbox_* / crop_* / detected_at
confidence / bbox_* / created_at image_data (BYTEA)   image_data (BYTEA)
```

**pgvector 확장**: `face_images.embedding_vec vector(512)` + HNSW 코사인 인덱스 (`init.sql`)

---

## 8. 모델 구성

| 기능 | 모델 파일 | 백엔드 |
|------|-----------|--------|
| 얼굴 임베딩 | ArcFace (deepface 내장) | TensorFlow / RetinaFace 검출 |
| 얼굴 인메모리 검색 | FAISS `IndexFlatIP` (코사인) | CPU |
| 사람 감지·추적 | `models/4people-yolo26n.pt` + ByteTrack | CUDA:0 |
| 동물 감지 | `models/best-4animals-yolo26m-hpo.pt` | CUDA:0 |
| 식물 병해 탐지 | `models/best-4plants-fasterRCNN.pt` (ConvNeXt+FPN) | CUDA:0 |
| 상추 병해 탐지 | `models/best_model_lettuce.pt` (콘보루션널 분류) | CUDA:0 |

---

## 9. 설정 파일 (.env 권장 값)

```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@<HOST>:5555/cctv?sslmode=disable

# FastAPI 공개 IP — 동물/식물 탐지 이미지 URL 생성 및 Streamlit API 호출에 사용
# 미설정 시: Streamlit은 localhost 폴백, 이미지 URL은 외부 전송 payload에서 생략
FASTAPI_PUBLIC_HOST=http://<PUBLIC_IP>:8000

PERSON_VIDEO_SOURCE=0
PERSON_CAMERA_ID=cam_01

# 멀티카메라 사용 시 (CAMERAS_JSON 우선 적용)
# CAMERAS_JSON=[{"camera_id":"cam_01","video_source":"0","label":"정문"}]

RESULT_PUBLISHER_BASE_URL=http://172.16.15.43:8000
FACE_DB_PATH=face_db
```

---

## 10. 빠른 시작

```bash
# 1. 의존성 설치
uv sync

# 2. PostgreSQL + pgvector 준비 후 init.sql 실행
psql -h <HOST> -p 5555 -U postgres -d cctv -f init.sql

# 3. FastAPI 서버
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000

# 4. Streamlit UI (선택)
uv run streamlit run app.py --server.port 8501

# 5. API 문서
# http://localhost:8000/docs
```

---

## 11. 프로젝트 디렉터리 구조

```
deepface_live/
├── server/                      # FastAPI 서버
│   ├── main.py                  # lifespan + 라우터 등록
│   ├── config.py                # pydantic-settings (.env 로드)
│   ├── database.py              # SQLAlchemy 엔진/세션
│   ├── models.py                # ORM 모델
│   ├── schemas.py               # Pydantic 요청/응답 스키마
│   ├── repositories/            # DB 접근 계층
│   │   ├── base.py
│   │   └── postgres_repo.py
│   ├── routers/                 # API 엔드포인트
│   │   ├── animal.py
│   │   ├── log.py
│   │   ├── person.py
│   │   ├── person_count.py      # 멀티카메라 인원 카운팅
│   │   ├── plant.py
│   │   └── recognition.py
│   └── services/
│       ├── animal/animal_service.py
│       ├── common/
│       │   ├── line_tracker.py
│       │   └── result_publisher.py
│       ├── face/face_service.py
│       ├── person/
│       │   ├── camera_config.py  # 카메라 1대 설정 dataclass
│       │   ├── camera_manager.py # N대 통합 관리자
│       │   ├── detector.py
│       │   └── person_service.py
│       └── plant/
│           ├── lettuce_service.py
│           └── plant_service.py
├── app.py                       # Streamlit 관리자 UI
├── lettuce_checkpoint_loader.py # 상추 모델 체크포인트 로더 (개발용)
├── lettuce_infer.ipynb         # 상추 탐지 노트북 (개발용)
├── pages/
│   └── animal_crossing_webcam.py # WebRTC 실시간 동물 탐지
├── tests/                       # pytest 자동화 테스트
│   ├── conftest.py
│   ├── test_animal_service.py
│   ├── test_face_service.py
│   ├── test_person_api.py
│   ├── test_plant_predict.py
│   ├── test_recognition_api.py
│   └── test_repository.py
├── scripts/
│   └── diag/                    # 개발 중 진단 스크립트 (pytest 대상 아님)
│       ├── diag_arcface_distinction.py
│       ├── diag_arcface_verify.py
│       ├── diag_embedding_diag.py
│       ├── diag_embedding_diag2.py
│       └── diag_model_compare.py
├── models/                      # AI 모델 가중치 파일
├── face_db/                     # 등록된 얼굴 이미지
├── logs/                        # 서버 로그
├── _check_gpu.py                # GPU/TensorFlow 진단 스크립트
├── logger.py                    # 로깅 설정
├── init.sql                     # PostgreSQL 초기화 SQL
└── pyproject.toml
```
