# DeepFace Live

CCTV 기반 실시간 탐지 시스템 — 얼굴 인식, 동물 탐지, 식물(병해) 탐지를 통합 제공합니다.

---

## 구성 요소

| 컴포넌트 | 역할 |
|---|---|
| **FastAPI** (`server/`) | 추론·DB·외부 전송 REST API 서버 (포트 8000) |
| **Streamlit** (`app.py`) | 관리자 웹 UI (포트 8501) |
| **PostgreSQL** | 탐지 로그·인물·얼굴 임베딩 저장 |

---

## 빠른 시작

```bash
# 의존성 설치 (uv 사용)
uv sync

# .env 설정 (아래 환경변수 섹션 참조)
cp .env.example .env   # 없으면 직접 생성

# FastAPI 서버
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

# Streamlit UI
python -m streamlit run app.py --server.port 8501
```

---

## 환경변수 (`.env`)

| 키 | 기본값 | 설명 |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://...` | PostgreSQL 연결 문자열 |
| `FASTAPI_PUBLIC_HOST` | `""` | 탐지 이미지 URL에 사용할 공개 IP (`http://x.x.x.x:8000`). 비어 있으면 `request.base_url` 사용 |
| `RESULT_PUBLISHER_BASE_URL` | `http://172.16.15.43:8000` | 외부 메인 서버 주소 |
| `RESULT_PUBLISHER_TIMEOUT` | `2.0` | 외부 전송 타임아웃(초) |
| `FACE_DB_PATH` | `face_db` | 얼굴 이미지 저장 디렉터리 |
| `DEEPFACE_MODEL` | `ArcFace` | 얼굴 임베딩 모델 |
| `RECOGNITION_THRESHOLD` | `0.40` | 얼굴 인식 cosine 거리 임계값 |
| `PERSON_VIDEO_SOURCE` | `0` | 인원 카운팅 카메라 소스 |
| `CAMERAS_JSON` | `""` | 멀티카메라 JSON 배열 (비어 있으면 `PERSON_*` 단일 설정 사용) |
| `ENABLE_RECOGNITION_LOG` | `true` | 얼굴 인식 로그 DB 저장 여부 |

---

## API 엔드포인트

### 얼굴 인식 · 등록
| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/api/recognize` | 이미지 → 얼굴 인식 결과 반환 |
| `POST` | `/api/register` | 이미지 → 얼굴 등록 (person_id 없으면 자동 생성) |

### 인물 관리
| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/persons` | 전체 인물 목록 |
| `GET` | `/api/persons/{id}` | 인물 상세 |
| `PUT` | `/api/persons/{id}` | 인물 정보 수정 |
| `DELETE` | `/api/persons/{id}` | 인물 삭제 |

### 동물 탐지
| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/animal/status` | 모델 로드 상태 |
| `POST` | `/api/animal/detect` | 이미지 업로드 → 탐지 + DB 저장 + 외부 전송 |
| `POST` | `/api/animal/logs/batch` | 탐지 결과 배치 저장 (동영상 처리용) |
| `GET` | `/api/animal/logs` | 탐지 로그 조회 |
| `GET` | `/api/animal/logs/{id}/image` | 탐지 크롭 이미지 다운로드 |
| `POST` | `/api/animal/reset` | 라인 트래커 초기화 |

### 식물 탐지
| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/plant/status` | 모델 로드 상태 |
| `POST` | `/api/plant/detect` | 이미지 업로드 → 탐지 + DB 저장 + 외부 전송 |
| `GET` | `/api/plant/logs` | 탐지 로그 조회 |
| `GET` | `/api/plant/logs/{id}/image` | 탐지 크롭 이미지 다운로드 |

### 인원 카운팅
| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/api/count/status` | 카운터 상태 |
| `GET` | `/api/count/current` | 현재 인원 수 |
| `POST` | `/api/count/camera/pause` | 카메라 점유 해제 |
| `POST` | `/api/count/camera/resume` | 카메라 재개 |

---

## 외부 서버 전송 (Result Publisher)

탐지 이벤트 발생 시 `RESULT_PUBLISHER_BASE_URL`로 JSON을 POST합니다.  
전송 실패 시 로그만 남기고 서비스는 계속 동작합니다.

### 동물 탐지 — `POST /api/detections/animal`
```json
{
  "source": "cam-center-01",
  "class_name": "dog",
  "confidence": 0.9123,
  "bbox_x1": 120.5,
  "bbox_y1": 80.0,
  "bbox_x2": 340.0,
  "bbox_y2": 260.5,
  "detected_at": "2026-04-07T14:30:00+09:00",
  "detect_image_url": "http://172.16.30.124:8000/api/animal/logs/42/image"
}
```

### 식물 탐지 — `POST /api/detections/plant`
```json
{
  "source": "cam-center-01",
  "class_name": "strawberry_disease",
  "disease_code": 2,
  "disease_label": "흰가루병",
  "confidence": 0.8741,
  "bbox_x1": 50.0, "bbox_y1": 60.0, "bbox_x2": 200.0, "bbox_y2": 210.0,
  "crop_type": 0,
  "crop_name": "딸기",
  "detected_at": "2026-04-07T14:30:00+09:00",
  "detect_image_url": "http://172.16.30.124:8000/api/plant/logs/17/image"
}
```

### 얼굴 인식 — `POST /api/events`
```json
{
  "event_type": "face_recognition",
  "camera_id": "cam-center-01",
  "timestamp": "2026-04-07T14:30:00+09:00",
  "payload": {
    "person_id": 3,
    "person_name": "person3",
    "display_name": "홍길동",
    "confidence": 0.312,
    "bbox": [100, 80, 250, 260]
  }
}
```

---

## DB 스키마

### `animal_detection_logs`
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | BIGSERIAL | PK |
| `source` | VARCHAR(50) | 탐지 소스 (`api`/`webcam`/`video`) |
| `class_name` | VARCHAR(100) | 탐지 클래스 |
| `confidence` | FLOAT | 신뢰도 |
| `bbox_x1~y2` | FLOAT | 바운딩 박스 |
| `image_data` | BYTEA | 탐지 영역 크롭 이미지 |
| `detected_at` | TIMESTAMPTZ | 탐지 시각 (KST) |

### `plant_detection_logs`
`animal_detection_logs`와 동일한 이미지/시각 컬럼 + 식물 메타 (`disease_code`, `crop_type`, `shooting_type`, `grow_stage`, `area` 등)

### `persons` / `face_images`
인물 정보 및 얼굴 이미지 경로·ArcFace 임베딩(BYTEA + pgvector)

---

## 모델

| 파일 | 용도 |
|---|---|
| `models/4people-yolo26n.pt` | 인원 카운팅 YOLO |
| `models/best-4animals-yolo26m-hpo.pt` | 동물 탐지 YOLO |
| `models/best-4plants-fasterRCNN.pt` | 식물 병해 탐지 Faster R-CNN |

---

## 주요 의존성

- Python 3.12+
- PyTorch (CUDA 12.8) + Ultralytics / torchvision
- FastAPI + Uvicorn
- Streamlit 1.56 + streamlit-webrtc
- DeepFace (ArcFace, RetinaFace)
- SQLAlchemy + psycopg2 + pgvector
- OpenCV
