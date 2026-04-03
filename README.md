# DeepFace Live — 실시간 얼굴 인식 시스템

실시간 영상 스트림에서 얼굴을 검출·인식하고, 등록된 인물과 매칭하여 알림을 제공하는 시스템입니다.

## 주요 기능

- **실시간 얼굴 인식** — 웹캠/RTSP 스트림에서 실시간 얼굴 검출 및 인물 매칭
- **얼굴 등록** — 단일/다중 각도 이미지로 인물 등록 (품질 검증 포함)
- **인물 관리** — 인물 CRUD, 표시 이름·연락처·메모 관리
- **알림 규칙** — 특정 인물 인식 시 토스트/사운드 알림
- **인식 로그** — 인식 이력 조회, 통계, 자동 정리

## 기술 스택

| 영역       | 기술                                     |
| ---------- | ---------------------------------------- |
| 언어       | Python 3.12+                             |
| API 서버   | FastAPI + Uvicorn                        |
| 프론트엔드 | Streamlit                                |
| 얼굴 인식  | DeepFace (Buffalo_L + ONNX Runtime CUDA) |
| ORM        | SQLAlchemy 2.0                           |
| DB         | MySQL 8.0 / Redis 7 (선택)               |
| 설정       | pydantic-settings + .env                 |

## 설치

### 1. 저장소 클론 및 가상환경

```bash
git clone <repository-url>
cd deepface_live

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Docker DB 실행

```bash
# MySQL 백엔드
docker compose --profile mysql up -d

# Redis 백엔드
docker compose --profile redis up -d
```

### 3. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다:

```env
DB_BACKEND=mysql

# MySQL
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=root
DB_NAME=deepface_live

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Face Recognition
DEEPFACE_MODEL=Buffalo_L
DEEPFACE_DETECTOR=retinaface
RECOGNITION_THRESHOLD=0.40
```

### 4. 서버 실행

```bash
# FastAPI 서버
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit UI (별도 터미널)
streamlit run ui/app.py
```

## API 문서

서버 실행 후 Swagger UI에서 전체 API를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 주요 엔드포인트

| Method   | Path                        | 설명                         |
| -------- | --------------------------- | ---------------------------- |
| `POST`   | `/api/persons`              | 인물 생성                    |
| `GET`    | `/api/persons`              | 인물 목록                    |
| `GET`    | `/api/persons/{id}`         | 인물 상세 (얼굴 이미지 포함) |
| `PUT`    | `/api/persons/{id}`         | 인물 수정                    |
| `DELETE` | `/api/persons/{id}`         | 인물 삭제                    |
| `POST`   | `/api/recognize`            | 이미지 → 얼굴 인식           |
| `POST`   | `/api/register`             | 이미지 → 얼굴 등록           |
| `POST`   | `/api/register/multi-angle` | 다중 각도 일괄 등록          |
| `GET`    | `/api/logs`                 | 인식 로그 조회               |
| `GET`    | `/api/logs/stats`           | 인식 통계                    |
| `DELETE` | `/api/logs/cleanup`         | 오래된 로그 정리             |

## 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 커버리지 리포트
pytest tests/ --cov=server --cov-report=term-missing
```

## 프로젝트 구조

```
deepface_live/
├── server/
│   ├── main.py                  # FastAPI 앱 + lifespan
│   ├── config.py                # pydantic-settings 설정
│   ├── database.py              # SQLAlchemy 엔진/세션
│   ├── models.py                # ORM 모델
│   ├── schemas.py               # Pydantic 요청/응답 스키마
│   ├── redis_client.py          # Redis 클라이언트
│   ├── repositories/
│   │   ├── base.py              # AbstractRepository (5개 sub-repo)
│   │   ├── mysql_repo.py        # MySQL 구현체
│   │   └── redis_repo.py        # Redis 구현체
│   ├── routers/
│   │   ├── person.py            # 인물 CRUD API
│   │   ├── recognition.py       # 인식/등록 API
│   │   └── log.py               # 로그 API
│   └── services/
│       ├── face_service.py      # DeepFace 얼굴 서비스
│       └── alert_service.py     # 알림 서비스
├── ui/
│   ├── app.py                   # Streamlit 메인
│   ├── components/              # UI 컴포넌트
│   └── pages/                   # Streamlit 페이지
├── tests/
│   ├── conftest.py              # 테스트 Fixture
│   ├── test_person_api.py       # 인물 API 테스트
│   ├── test_recognition_api.py  # 인식/등록 API 테스트
│   ├── test_face_service.py     # FaceService 단위 테스트
│   └── test_repository.py       # Repository 인터페이스 테스트
├── face_db/                     # 등록 얼굴 이미지 저장소
├── logs/                        # 로그 파일
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 환경변수 목록

| 변수                           | 기본값          | 설명                              |
| ------------------------------ | --------------- | --------------------------------- |
| `DB_BACKEND`                   | `mysql`         | DB 백엔드 (`mysql` / `redis`)     |
| `DB_HOST`                      | `localhost`     | MySQL 호스트                      |
| `DB_PORT`                      | `3306`          | MySQL 포트                        |
| `DB_USER`                      | `root`          | MySQL 사용자                      |
| `DB_PASSWORD`                  | `root`          | MySQL 비밀번호                    |
| `DB_NAME`                      | `deepface_live` | MySQL 데이터베이스명              |
| `REDIS_HOST`                   | `localhost`     | Redis 호스트                      |
| `REDIS_PORT`                   | `6379`          | Redis 포트                        |
| `REDIS_PASSWORD`               | (빈 문자열)     | Redis 비밀번호                    |
| `FACE_DB_PATH`                 | `face_db`       | 얼굴 이미지 저장 경로             |
| `DEEPFACE_MODEL`               | `Buffalo_L`     | DeepFace 인식 모델 (ONNX GPU)     |
| `DEEPFACE_DETECTOR`            | `retinaface`    | 등록용 얼굴 검출기                |
| `DEEPFACE_DETECTOR_REALTIME`   | `retinaface`    | 실시간 검출기                     |
| `RECOGNITION_THRESHOLD`        | `0.40`          | 인식 임계값 (cosine distance)     |
| `FACE_MIN_CONFIDENCE`          | `0.90`          | 등록 시 최소 감지 신뢰도          |
| `FACE_MIN_SIZE`                | `112`           | 등록 시 최소 얼굴 크기 (px)       |
| `FACE_BLUR_THRESHOLD`          | `100.0`         | 등록 시 선명도 임계값 (Laplacian) |
| `FACE_MIN_CONFIDENCE_REALTIME` | `0.80`          | 실시간 최소 신뢰도                |
| `FACE_MIN_SIZE_REALTIME`       | `56`            | 실시간 최소 얼굴 크기 (px)        |
| `LOG_RETENTION_DAYS`           | `30`            | 로그 보관 일수                    |
| `LOG_DEDUP_SECONDS`            | `10`            | 인식 로그 중복 방지 간격 (초)     |

## 라이선스

MIT License
