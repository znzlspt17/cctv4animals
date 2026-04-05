# 실시간 사람 추적 카운팅 시스템 기획서

## 1. 프로젝트 개요

OpenCV + YOLOv8 + ByteTrack 기반의 **실시간 사람 추적 및 카운팅 시스템**.
카메라 영상에서 사람을 감지하고, 설정된 기준선을 통과할 때 IN/OUT을 카운팅한다.

---

## 2. 시스템 아키텍처

```
카메라 (RTSP / 웹캠 / MP4)
    │
    ▼
[OpenCV VideoCapture] ← Frame Skip (매 N프레임)
    │
    ▼
[YOLOv8 Detection] ← CUDA GPU 가속
    │  (bounding boxes)
    ▼
[ByteTrack Tracker] ← ID 할당 및 유지
    │  (tracked objects + IDs)
    ▼
[Line Crossing Logic] ← 중심좌표 y값 변화 감지
    │  (IN/OUT 이벤트)
    ├──▶ [MySQL DB 저장]
    ├──▶ [카카오톡 알림] ← 임계값 초과 시
    ├──▶ [스냅샷 저장] ← 이벤트 시점 이미지
    └──▶ [웹 대시보드] ← FastAPI + WebSocket
```

---

## 3. 주요 기능 스펙

### 3.1 객체 검출 및 추적

| 항목          | 상세                                    |
| ------------- | --------------------------------------- |
| 검출 모델     | YOLOv8 (ultralytics)                    |
| GPU 가속      | CUDA 활용                               |
| 추적 알고리즘 | ByteTrack (supervision 라이브러리 내장) |
| 대상 클래스   | person (사람만 필터링)                  |

### 3.2 카운팅 로직

- 화면에 설정된 **기준선**을 기준으로 판단
- Bounding Box의 **중심좌표(cx, cy)** 사용
- 선을 **위로 지나가면 +1** (IN), **아래로 지나가면 -1** (OUT)
- 대각선 기준선 지원 (카메라 각도 대응)

### 3.3 프레임 건너뛰기 (Frame Skip)

- 30fps 영상 기준, 매 프레임 추론은 GPU 과부하
- **10프레임마다 YOLO 검출**, 나머지 프레임은 ByteTrack 예측(칼만 필터)만 수행
- Frame Skip 간격은 `.env` 설정으로 관리

### 3.4 중복 카운팅 방지

- `counted_ids = set()` 으로 동일 ID 1회만 카운트
- 일정 시간 경과 후 set에서 제거 (재진입 허용 시)
- ID별 마지막 라인 통과 방향 기록

### 3.5 ROI (Region of Interest)

- 전체 프레임이 아닌 **특정 영역만 검출** 대상
- 불필요한 영역 제외 → 오탐 감소 + 성능 향상
- ROI 좌표는 설정 파일로 관리

### 3.6 영상 소스

- **웹캠**: `cv2.VideoCapture(0)`
- **RTSP 스트림**: `cv2.VideoCapture("rtsp://...")`
- **MP4 파일**: `cv2.VideoCapture("video.mp4")`
- 통합 인터페이스로 소스 종류에 관계없이 동일 파이프라인 처리

---

## 4. 카운팅 라인 설정 UI

- 마우스 드래그로 카운팅 라인 **위치/각도** 설정
- 수평선 및 **대각선** 지원
- 좌/우 또는 상/하 **디폴트 라인** 제공
- 설정값 `.env` 파일에 저장

---

## 5. DB 설계 (MySQL)

### 5.1 tracking_events 테이블

| 컬럼             | 타입                   | 설명                    |
| ---------------- | ---------------------- | ----------------------- |
| `id`             | INT AUTO_INCREMENT     | PK                      |
| `person_id`      | INT                    | ByteTrack 할당 ID       |
| `camera_id`      | VARCHAR(50)            | 카메라 식별자           |
| `direction`      | ENUM('IN', 'OUT')      | 이동 방향 (위로/아래로) |
| `count_change`   | INT                    | +1 또는 -1              |
| `current_count`  | INT                    | 시점별 총 인원수        |
| `confidence`     | FLOAT                  | 검출 신뢰도             |
| `bbox_x`         | INT                    | 바운딩박스 x좌표        |
| `bbox_y`         | INT                    | 바운딩박스 y좌표        |
| `bbox_w`         | INT                    | 바운딩박스 너비         |
| `bbox_h`         | INT                    | 바운딩박스 높이         |
| `tracking_start` | DATETIME               | 추적 시작 시간          |
| `tracking_end`   | DATETIME               | 추적 종료 시간          |
| `snapshot_path`  | VARCHAR(255)           | 스냅샷 파일 경로        |
| `created_at`     | DATETIME DEFAULT NOW() | DB 기록 시점            |

### 5.2 활용

- 시점별 인원 추이 분석 (`current_count`)
- 오탐 필터링 (`confidence` 기준)
- 디버깅/분석 (`bbox` 좌표)
- 이벤트 추적 (`snapshot_path`)

---

## 6. 알림 시스템 (카카오톡)

- 현재 카운트가 **설정 임계값 초과** 시 알림 발생
- 카카오 REST API ("나에게 보내기") 사용
- 알림 쿨다운 설정 (동일 알림 반복 방지, 예: 5분 간격)
- 카카오 디벨로퍼 앱 등록 및 토큰 갱신 자동화 필요

---

## 7. 웹 대시보드

- **FastAPI + WebSocket** 기반 실시간 모니터링
- 표시 항목:
  - 현재 인원수
  - 누적 IN / OUT 카운트
  - 카메라별 현황
  - 실시간 영상 스트리밍 (옵션)

---

## 8. 성능 모니터링

| 지표       | 측정 방법                | 표시 위치                |
| ---------- | ------------------------ | ------------------------ |
| FPS (전체) | 프레임 처리 시간 역수    | 영상 오버레이 + 대시보드 |
| 추론 시간  | YOLO inference 소요 ms   | 대시보드                 |
| 추적 시간  | ByteTrack update 소요 ms | 대시보드                 |
| GPU 메모리 | `pynvml` 라이브러리      | 대시보드                 |

### 예상 FPS (RTX 3060 기준)

| 모델            | Frame Skip 없음 | Frame Skip 10    |
| --------------- | --------------- | ---------------- |
| YOLOv8n (nano)  | 25~40 fps       | 60+ fps (효과적) |
| YOLOv8s (small) | 15~25 fps       | 40+ fps (효과적) |

---

## 9. 녹화 및 스냅샷

- 카운트 변화 시점의 프레임 **스냅샷 이미지 저장** (증거용)
- 이벤트 발생 전후 **N초 클립 저장** (옵션)
- 파일 경로를 DB `snapshot_path` 컬럼에 기록

---

## 10. 재시작 시 상태 복구

- 시스템 재시작 시 **마지막 카운트값을 DB에서 복원**
- 추적 중이던 ID 목록 저장/복구
- 안정적인 장기 운영 보장

---

## 11. 기술 스택

### 핵심 라이브러리

| 역할             | 라이브러리               | 버전              | 비고                                 |
| ---------------- | ------------------------ | ----------------- | ------------------------------------ |
| 객체 검출        | `ultralytics`            | 8.3.x             | YOLOv8 공식, CUDA 자동 지원          |
| GPU 가속         | `torch` + `torchvision`  | 2.5.x + CUDA 12.4 | ultralytics 의존                     |
| 추적 (ByteTrack) | `supervision`            | 0.25.x            | Roboflow 통합 추적, ByteTrack 내장   |
| 영상 처리        | `opencv-python-headless` | 4.10.x            | CUDA 빌드 시 `opencv-contrib-python` |
| DB               | `SQLAlchemy` + `PyMySQL` | 2.0.x             | MySQL ORM                            |
| 웹 대시보드      | `FastAPI` + `uvicorn`    | 0.115.x           | WebSocket 지원                       |
| 카카오톡 알림    | `requests`               | 2.32.x            | 카카오 REST API                      |
| 설정 관리        | `python-dotenv`          | 1.0.x             | API 키 / DB URL / 라인 설정          |
| GPU 모니터링     | `pynvml`                 | 12.x              | GPU 메모리 사용량                    |
| 로깅             | `loguru`                 | 0.7.x             | 구조화된 로그                        |

### 설치

```bash
# PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 나머지 의존성
pip install ultralytics supervision opencv-python-headless sqlalchemy pymysql fastapi uvicorn requests python-dotenv pynvml loguru
```

---

## 12. 환경 설정 (.env)

```env
# 영상 소스
VIDEO_SOURCE=0                    # 0: 웹캠, rtsp://..., video.mp4
CAMERA_ID=cam_01

# 카운팅 라인 (디폴트)
LINE_START_X=0
LINE_START_Y=360
LINE_END_X=1280
LINE_END_Y=360
LINE_DIRECTION=horizontal         # horizontal / vertical / diagonal

# 프레임 건너뛰기
FRAME_SKIP=10

# ROI
ROI_X=0
ROI_Y=100
ROI_W=1280
ROI_H=520

# DB
DB_HOST=localhost
DB_PORT=3306
DB_NAME=people_counter
DB_USER=root
DB_PASSWORD=

# 알림
KAKAO_ACCESS_TOKEN=
KAKAO_REFRESH_TOKEN=
ALERT_THRESHOLD=50
ALERT_COOLDOWN_SEC=300

# YOLO
YOLO_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
```

---

## 13. 프로젝트 디렉터리 구조 (예정)

```
people_counter/
├── .env                     # 환경 설정
├── requirements.txt         # 의존성 목록
├── main.py                  # 메인 실행 파일
├── config.py                # .env 로딩 및 설정 클래스
├── detector.py              # YOLOv8 검출기
├── tracker.py               # ByteTrack 추적기
├── counter.py               # 라인 크로싱 카운터 로직
├── database.py              # SQLAlchemy 모델 및 DB 연결
├── alert.py                 # 카카오톡 알림
├── snapshot.py              # 스냅샷/녹화 관리
├── monitor.py               # FPS/GPU 성능 모니터링
├── dashboard/               # 웹 대시보드
│   ├── app.py               # FastAPI 앱
│   └── templates/           # HTML 템플릿
├── models/                  # YOLO 모델 파일
│   └── yolov8n.pt
└── snapshots/               # 스냅샷 저장 디렉터리
```

---

## 14. 라이브러리 호환성 검토 (Python 3.12 기준)

### 14.1 Python 3.12 호환 버전 매트릭스

| #   | 라이브러리    | PyPI 패키지명            | Python 3.12 지원 버전 | 권장 버전        | 3.12 호환 상태 | 비고                                              |
| --- | ------------- | ------------------------ | --------------------- | ---------------- | :------------: | ------------------------------------------------- |
| 1   | PyTorch       | `torch`                  | ≥ 2.2.0               | **2.5.1+cu124**  |  ✅ 정식 지원  | 2.1.x 이하는 3.12 미지원                          |
| 2   | TorchVision   | `torchvision`            | ≥ 0.17.0              | **0.20.1+cu124** |  ✅ 정식 지원  | torch 버전과 반드시 매칭                          |
| 3   | Ultralytics   | `ultralytics`            | ≥ 8.1.0               | **8.3.40**       |  ✅ 정식 지원  | 8.0.x는 3.12 wheel 없음                           |
| 4   | Supervision   | `supervision`            | ≥ 0.19.0              | **0.25.0**       |  ✅ 정식 지원  | ByteTrack 내장                                    |
| 5   | OpenCV        | `opencv-python-headless` | ≥ 4.9.0               | **4.10.0.84**    |  ✅ 정식 지원  | 4.8.x 이하 3.12 wheel 없음                        |
| 6   | NumPy         | `numpy`                  | ≥ 1.26.0              | **1.26.4**       |  ✅ 정식 지원  | ⚠️ 1.25.x 이하 3.12 미지원, 2.x는 torch 충돌 위험 |
| 7   | SQLAlchemy    | `sqlalchemy`             | ≥ 2.0.23              | **2.0.36**       |  ✅ 정식 지원  | C 확장 3.12 빌드 포함                             |
| 8   | PyMySQL       | `pymysql`                | ≥ 1.1.0               | **1.1.1**        |  ✅ 정식 지원  | pure-Python, 버전 제약 없음                       |
| 9   | FastAPI       | `fastapi`                | ≥ 0.104.0             | **0.115.6**      |  ✅ 정식 지원  | Pydantic v2 필수                                  |
| 10  | Uvicorn       | `uvicorn[standard]`      | ≥ 0.24.0              | **0.34.0**       |  ✅ 정식 지원  | `[standard]`로 websockets 포함                    |
| 11  | Requests      | `requests`               | ≥ 2.31.0              | **2.32.3**       |  ✅ 정식 지원  |                                                   |
| 12  | python-dotenv | `python-dotenv`          | ≥ 1.0.0               | **1.0.1**        |  ✅ 정식 지원  |                                                   |
| 13  | pynvml        | `pynvml`                 | ≥ 11.5.0              | **12.560.30**    |  ✅ 정식 지원  | NVIDIA 드라이버 의존                              |
| 14  | Loguru        | `loguru`                 | ≥ 0.7.0               | **0.7.3**        |  ✅ 정식 지원  |                                                   |
| 15  | Pillow        | `pillow`                 | ≥ 10.1.0              | **10.4.0**       |  ✅ 정식 지원  | ultralytics 의존성                                |
| 16  | Pydantic      | `pydantic`               | ≥ 2.5.0               | **2.10.3**       |  ✅ 정식 지원  | FastAPI 의존성                                    |

### 14.2 Python 3.12 주의사항

| #   | 항목                 | 내용                                                                     |
| --- | -------------------- | ------------------------------------------------------------------------ |
| 1   | **distutils 제거**   | Python 3.12에서 `distutils` 모듈이 완전 제거됨. `setuptools ≥ 69.0` 필요 |
| 2   | **numpy 최소 버전**  | numpy 1.25 이하는 3.12 wheel이 없어 빌드 실패 → **1.26.0 이상 필수**     |
| 3   | **torch 최소 버전**  | torch 2.1.x 이하는 3.12 미지원 → **2.2.0 이상 필수**                     |
| 4   | **opencv 최소 버전** | opencv-python 4.8.x 이하는 3.12 빌드 없음 → **4.9.0 이상 필수**          |
| 5   | **C 확장 재빌드**    | 일부 패키지 구버전이 3.12 ABI 변경으로 컴파일 실패 가능                  |

### 14.3 의존성 충돌 위험 및 해결

| 충돌 지점             | 위험도  | 원인                                                                 | 해결책                                         |
| --------------------- | :-----: | -------------------------------------------------------------------- | ---------------------------------------------- |
| **numpy 버전 범위**   | 🔴 높음 | torch 2.5 → numpy <2.5 요구, 일부 라이브러리가 numpy 2.x를 설치 시도 | `numpy==1.26.4` 핀 (1.x 최종판, 전체 호환)     |
| **opencv 중복 설치**  | 🟡 중간 | ultralytics가 `opencv-python` 설치, headless와 공존 불가             | headless 설치 후 `pip uninstall opencv-python` |
| **CUDA 버전 불일치**  | 🔴 높음 | torch cu124 빌드와 시스템 드라이버 CUDA 버전 불일치                  | `nvidia-smi` CUDA ≥ 12.4 확인 (드라이버 525+)  |
| **Pydantic v1 vs v2** | 🟡 중간 | FastAPI 0.115는 Pydantic v2 필수, 일부 구 패키지가 v1 요구           | Pydantic ≥ 2.5 사용, v1 호환 모드 불필요       |
| **websockets 누락**   | 🟢 낮음 | uvicorn만 설치 시 WebSocket 지원 안됨                                | `uvicorn[standard]`로 설치                     |

### 14.4 검증된 설치 순서 (Python 3.12)

```bash
# 0) Python 3.12 가상환경 생성
python -m venv venv
venv\Scripts\activate

# 1) setuptools 업데이트 (distutils 제거 대응)
pip install --upgrade pip setuptools>=69.0

# 2) PyTorch + CUDA 12.4 (가장 먼저 — numpy 버전 기준점)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3) numpy 버전 고정 (2.x 방지)
pip install numpy==1.26.4

# 4) ultralytics (YOLOv8)
pip install ultralytics==8.3.40

# 5) opencv headless로 교체
pip uninstall opencv-python -y
pip install opencv-python-headless==4.10.0.84

# 6) supervision (ByteTrack)
pip install supervision==0.25.0

# 7) DB + 웹 + 유틸리티
pip install sqlalchemy==2.0.36 pymysql==1.1.1
pip install fastapi==0.115.6 "uvicorn[standard]==0.34.0"
pip install requests==2.32.3 python-dotenv==1.0.1 pynvml loguru==0.7.3
```

### 14.5 설치 후 검증 스크립트

```python
# verify_env.py
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

import numpy as np
print(f"NumPy: {np.__version__}")

import cv2
print(f"OpenCV: {cv2.__version__}")

import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

import supervision
print(f"Supervision: {supervision.__version__}")

import sqlalchemy
print(f"SQLAlchemy: {sqlalchemy.__version__}")

import fastapi
print(f"FastAPI: {fastapi.__version__}")

import loguru
print(f"Loguru: OK")

# numpy 2.x 충돌 확인
assert int(np.__version__.split('.')[0]) == 1, "⚠️ numpy 2.x 감지 — torch 충돌 위험!"
print("\n✅ 전체 호환성 검증 통과")
```

### 14.6 requirements.txt

```
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.5.1
torchvision==0.20.1
numpy==1.26.4
ultralytics==8.3.40
opencv-python-headless==4.10.0.84
supervision==0.25.0
sqlalchemy==2.0.36
pymysql==1.1.1
fastapi==0.115.6
uvicorn[standard]==0.34.0
requests==2.32.3
python-dotenv==1.0.1
pynvml
loguru==0.7.3
```

---

## 15. 커스텀 에이전트 구성 계획

### 15.1 에이전트 개요

프로젝트의 기술 도메인(CV/ML, 백엔드, 프론트엔드, 설정/인프라, 테스트)별로 전문 에이전트를 만들어 역할을 분리하고, 각 에이전트가 관련 파일/도구에만 집중하도록 구성한다.

### 15.2 에이전트 목록

| #   | 에이전트 이름 | 역할                                      | 담당 파일                                                              | 핵심 도구                       |
| --- | ------------- | ----------------------------------------- | ---------------------------------------------------------------------- | ------------------------------- |
| 1   | **cv**        | 컴퓨터 비전 파이프라인 (검출/추적/카운팅) | `detector.py`, `tracker.py`, `counter.py`, `snapshot.py`, `monitor.py` | 파일 편집, 터미널 (YOLO 테스트) |
| 2   | **backend**   | FastAPI 서버, DB, 알림 시스템             | `database.py`, `alert.py`, `dashboard/app.py`, `config.py`             | 파일 편집, 터미널 (서버 실행)   |
| 3   | **frontend**  | 웹 대시보드 UI (HTML/JS/CSS)              | `dashboard/templates/**`                                               | 파일 편집, 브라우저 도구        |
| 4   | **devops**    | 환경 설정, 의존성, Docker, 배포           | `.env`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`         | 터미널, 파일 편집               |
| 5   | **tester**    | 테스트 작성 및 검증                       | `tests/**`                                                             | 터미널 (pytest), 파일 편집      |

### 15.3 에이전트 상세

#### 15.3.1 `cv` — 컴퓨터 비전 전문가

- **역할**: YOLOv8 검출, ByteTrack 추적, 라인 크로싱 카운팅 로직, 스냅샷 저장, 성능 모니터링
- **전문 지식**: `ultralytics`, `supervision` (ByteTrack), OpenCV, CUDA GPU 가속, 칼만 필터 기반 프레임 스킵
- **사용 시점**: 검출 정확도 개선, 추적 알고리즘 튜닝, ROI 설정, 중복 카운팅 방지 로직 구현
- **제한**: DB/웹 코드 직접 수정 금지 → 인터페이스만 정의

#### 15.3.2 `backend` — 백엔드 API/DB 전문가

- **역할**: FastAPI 앱, WebSocket 실시간 통신, SQLAlchemy ORM, 카카오톡 알림, 상태 복구
- **전문 지식**: FastAPI, SQLAlchemy, PyMySQL, WebSocket, REST API, 카카오 OAuth
- **사용 시점**: DB 스키마 설계, API 엔드포인트 구현, 알림 로직, 재시작 시 상태 복구

#### 15.3.3 `frontend` — 대시보드 UI 전문가

- **역할**: 웹 대시보드 HTML/JS/CSS, 실시간 데이터 표시, 차트/그래프
- **전문 지식**: Jinja2 템플릿, WebSocket 클라이언트, Chart.js 등 시각화
- **사용 시점**: 대시보드 레이아웃, 실시간 인원수 표시, 카메라별 현황 UI

#### 15.3.4 `devops` — 환경/인프라 전문가

- **역할**: Python 가상환경, CUDA 설정, Docker 컨테이너화, `.env` 관리
- **전문 지식**: pip, Docker, CUDA 드라이버, 의존성 충돌 해결
- **사용 시점**: 초기 환경 구축, 배포 설정, dependency 충돌 해결

#### 15.3.5 `tester` — 테스트 전문가

- **역할**: 단위 테스트, 통합 테스트, CV 파이프라인 검증
- **전문 지식**: pytest, mock 객체, 영상 테스트 데이터
- **사용 시점**: 카운팅 로직 정확성 검증, API 엔드포인트 테스트, 엣지 케이스 확인

### 15.4 구현 순서

1. **Phase 1**: `devops` → 환경 설정 및 프로젝트 구조 생성 (다른 모든 작업의 전제)
2. **Phase 2** (병렬 가능):
   - `cv` → 검출/추적/카운팅 핵심 로직
   - `backend` → DB 스키마 및 기본 API
3. **Phase 3**: `frontend` → 대시보드 UI (`backend` 완료 후)
4. **Phase 4**: `tester` → 전체 통합 테스트

### 15.5 추가 고려사항

1. **에이전트 수 조절**: 프로젝트 규모가 크지 않다면 `cv` + `backend`(프론트 포함) + `devops` **3개로 축소**하는 것도 가능
2. **공통 규칙**: 모든 에이전트가 따를 프로젝트 공통 규칙(코딩 컨벤션, 로깅 패턴, `.env` 사용법)은 `.github/copilot-instructions.md`에 정의
3. **`main.py` 통합자**: `main.py`는 모든 모듈을 조합하는 진입점이므로, 특정 에이전트 전담보다는 기본 에이전트나 사용자가 직접 관리
