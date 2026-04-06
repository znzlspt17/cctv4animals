# DeepFace Live — 프로젝트 현황 보고서

> 작성일: 2026-04-07  
> 기준 브랜치: `main` (로컬)

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
5. CameraManager.start_all() → cam_01 스레드 시작
6. AnimalService warmup (YOLO)
7. PlantService warmup (FasterRCNN)
```

---

## 2. 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────┐
│  Streamlit (app.py :8501)                        │
│  · 얼굴 등록/인식  · 동물/식물 탐지  · 라인 설정  │
│  pages/animal_crossing_webcam.py (WebRTC 실시간) │
└──────────────────┬───────────────────────────────┘
                   │ HTTP
┌──────────────────▼───────────────────────────────┐
│  FastAPI (server/main.py :8000)                  │
│                                                  │
│  /api/recognize   /api/register   /api/persons   │
│  /api/count/*     /api/animal/*   /api/plant/*   │
│  /api/logs/*                                     │
│                                                  │
│  ┌──────────────┐  ┌────────────────────────┐   │
│  │  FaceService │  │    CameraManager       │   │
│  │  ArcFace 512 │  │  cam_01 (source="0")   │   │
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

## 3. 서버 역할 및 입력 모델 (아키텍처 설계 의도)

### 이 서버의 실제 역할

이 서버는 **AI 엣지 서버**로, 카메라 스트림 제어가 목적이 아니라 다음 세 가지 역할에 집중합니다:

1. **AI 추론**: 영상 프레임에서 사람·얼굴·동물·식물 분석
2. **결과 저장**: PostgreSQL (pgvector 포함) 에 추론 결과 적재
3. **이벤트 전송**: 메인 서버(`172.16.15.43:8000`)에 탐지 이벤트 푸시

```
┌──────────────────────────────────┐
│  OS / 하드웨어 전용 소프트웨어    │
│  (카메라 컨트롤 전담)             │
│  · RTSP / HLS / MJPEG 스트림 출력│
└────────────────┬─────────────────┘
                 │ 스트림 or 프레임 HTTP POST
┌────────────────▼─────────────────┐
│   이 서버 (AI 엣지 서버)          │
│   · AI 추론 (YOLO / ArcFace 등)  │
│   · PostgreSQL 저장               │
│   · 메인 서버 이벤트 전송         │
└────────────────┬─────────────────┘
                 │ publish_event()
┌────────────────▼─────────────────┐
│  메인 서버 172.16.15.43:8000      │
│  (비즈니스 로직 / UI / 알림 등)   │
└──────────────────────────────────┘
```

### 카메라 입력 모델 — 데모 vs 운영

| 항목 | 현재 (데모) | 실제 운영 |
|------|-------------|-----------|
| 카메라 접근 | `cv2.VideoCapture(0)` 직접 열기 | OS/전용 SW가 RTSP 스트림 제공 |
| 프레임 공급 | `PersonCounterService` 내부 루프 | 외부 소프트웨어 → RTSP or HTTP POST |
| 얼굴·동물·식물 | `POST /api/recognize` 등으로 외부에서 프레임 전송 | 동일 방식 유지 or 스트림 구독으로 전환 |
| 제어 권한 | 이 서버가 카메라 0번 점유 | OS 레이어에서 카메라 독점 관리 |

> **결론**: `cv2.VideoCapture` 루프 및 HTTP POST 방식 모두 데모 수준의 **입력 레이어**이며,  
> 추론 → DB 저장 → 이벤트 전송 **파이프라인**이 이 서버의 실제 핵심입니다.

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
CAMERAS_JSON=[
  {"camera_id":"cam_01","video_source":"0","label":"정문"},
  {"camera_id":"cam_02","video_source":"rtsp://192.168.1.10/stream","label":"후문",
   "line_start_y":540,"line_end_y":540}
]
```

**런타임 동적 추가 (서버 실행 중)**
```http
POST /api/count/cameras
Content-Type: application/json
{"camera_id":"cam_03","video_source":"1","label":"측문"}
```

### 멀티카메라 API 목록

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
```

---

## 5. 현재 이슈 / 알려진 제한사항

### ✅ 해결됨

| 항목 | 내용 | 해결 |
|------|------|------|
| `person_counter_service` import 오류 | `__init__.py`가 삭제된 싱글톤을 임포트 | `CameraManager`/`CameraConfig`로 교체 |
| `recognition.py` 타입 오류 | `person_id: int \| None`을 `RegisterResponse(person_id=...)` 에 직접 전달 | `assert person_id is not None` 추가 |

### ⚠️ 주의 / 잠재 이슈

| 항목 | 수준 | 내용 |
|------|------|------|
| PersonDetector 하드코딩 | **중** | `detector.py`가 여전히 `settings.PERSON_ROI_*`를 전역 settings에서 읽음. 카메라별 ROI는 `CameraConfig`에 필드는 있으나 `PersonDetector`에 미전달 |
| 외부 이벤트 서버 미응답 | **낮음** | `RESULT_PUBLISHER_BASE_URL=172.16.15.43:8000` 미응답 시 로그 경고만 발생 (비차단) |
| DB 세션 컨텍스트 매니저 | **중** | `get_events()`에서 `SessionLocal()`이 반환된 ORM 객체를 세션 외부에서 사용 (`LazyLoad` 안전하지 않음) — 현재는 단순 컬럼만 접근하므로 동작하나, 관계 필드 접근 시 `DetachedInstanceError` 가능 |
| 카메라 0번 단일 점유 | **중** | `cam_01`이 USB 카메라 `0`번을 상시 점유. Streamlit WebRTC 페이지 사용 시 반드시 `POST /api/count/camera/pause` 선행 필요 |
| `face_image` `face_image_id` 미반환 | **낮음** | `get_all_embeddings()`가 `face_image_id`를 반환하지 않아 캐시 항목의 `face_image_id=None` 가능 |

### ❌ 미구현

| 항목 | 내용 |
|------|------|
| 카메라별 PersonDetector ROI | `CameraConfig.roi_*` 필드 존재, `PersonDetector`에 파라미터 전달 미구현 |
| 스냅샷 저장 | `snapshot_path = ""` 하드코딩 — 탐지 이벤트 이미지 저장 불가 |
| 인식 로그 중복 제거 | `is_duplicate_log()` 구현됨, 라우터에서 미호출 |
| 동물 라인 크로싱 DB 저장 | Streamlit WebRTC 페이지에서 이벤트 탐지 후 DB 저장 미구현 |

---

## 6. 데이터베이스 스키마 요약

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

tracking_events                  animal_detection_logs
──────────────────────────       ──────────────────────
id (BigInt PK)                   id / source / class_name
camera_id ← 멀티카메라 구분       confidence / bbox_* / detected_at
tracker_id / direction
count_change / current_count
confidence / bbox_* / created_at
```

**pgvector 확장**: `face_images.embedding_vec vector(512)` + HNSW 코사인 인덱스 (`init.sql`)

---

## 7. 모델 구성

| 기능 | 모델 파일 | 백엔드 |
|------|-----------|--------|
| 얼굴 임베딩 | ArcFace (deepface 내장) | TensorFlow / RetinaFace 검출 |
| 얼굴 인메모리 검색 | FAISS `IndexFlatIP` (코사인) | CPU |
| 사람 감지·추적 | `models/4people-yolo26n.pt` + ByteTrack | CUDA:0 |
| 동물 감지 | `models/best-4animals-yolo26m-hpo.pt` | CUDA:0 |
| 식물 병해 탐지 | `models/best-4plants-fasterRCNN.pt` (ConvNeXt+FPN) | CUDA:0 |

---

## 8. 설정 파일 (.env 권장 값)

```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@<HOST>:5555/cctv?sslmode=disable

# 단일 카메라 (기본)
PERSON_VIDEO_SOURCE=0
PERSON_CAMERA_ID=cam_01

# 멀티카메라 (CAMERAS_JSON 우선 적용)
# CAMERAS_JSON=[{"camera_id":"cam_01","video_source":"0","label":"정문"}]

RESULT_PUBLISHER_BASE_URL=http://172.16.15.43:8000
FACE_DB_PATH=face_db
```

---

## 9. 빠른 시작

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
