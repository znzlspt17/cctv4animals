# 실시간 사람 추적 카운팅 시스템 — 상세 워크플로우

> 각 Phase별 담당 에이전트와 Step별 구현 내용, 의존 관계, 검증 기준을 정의한다.
> 기획서 `people_counter_plan.md`의 §15를 기반으로 작성.

---

## Phase 1: 프로젝트 초기화 (devops)

> 모든 작업의 전제 조건. 이 Phase 완료 전 다른 에이전트 작업 불가.

### Step 1.1 — 프로젝트 디렉터리 구조 생성

- **담당**: `devops`
- **선행 조건**: 없음
- **작업 내용**:
  - 기획서 §13의 디렉터리 구조대로 폴더/빈 파일 생성
  - 생성 대상:
    - `main.py`, `config.py`, `detector.py`, `tracker.py`, `counter.py`
    - `database.py`, `alert.py`, `snapshot.py`, `monitor.py`
    - `dashboard/app.py`, `dashboard/templates/`
    - `models/`, `snapshots/`, `tests/`
- **✅ 검증**: 모든 파일/폴더 존재 확인

### Step 1.2 — 환경 설정 파일 작성

- **담당**: `devops`
- **선행 조건**: Step 1.1
- **작업 내용**:
  - `.env` 파일 생성 (기획서 §12의 전체 설정값 포함)
  - `.env.example` 복사본 생성 (민감 정보 빈칸)
  - `.gitignore` 생성 (`.env`, `venv/`, `__pycache__/`, `snapshots/`, `models/*.pt` 제외)
- **✅ 검증**: `.env` 로딩 테스트 (`python -c "from dotenv import load_dotenv; load_dotenv(); print('OK')"`)

### Step 1.3 — Python 가상환경 및 의존성 설치

- **담당**: `devops`
- **선행 조건**: Step 1.2
- **작업 내용**:
  - 기획서 §14.4의 검증된 설치 순서 그대로 실행
  - `requirements.txt` 파일 정리 (§14.6 내용)
  - opencv 중복 설치 문제 해결 (`pip uninstall opencv-python -y`)
- **✅ 검증**: `verify_env.py` (§14.5) 실행하여 전체 호환성 통과 확인

### Step 1.4 — config.py 작성

- **담당**: `devops`
- **선행 조건**: Step 1.3
- **작업 내용**:
  - `python-dotenv`로 `.env` 로딩
  - 모든 설정값을 담는 `Settings` 클래스 (dataclass 또는 Pydantic BaseSettings)
  - 설정 항목: `VIDEO_SOURCE`, `CAMERA_ID`, `LINE_*`, `FRAME_SKIP`, `ROI_*`, `DB_*`, `KAKAO_*`, `ALERT_*`, `YOLO_*`, `CONFIDENCE_THRESHOLD`
  - 타입 변환 포함 (문자열 → int/float)
  - 싱글톤 `settings` 인스턴스 export
- **✅ 검증**: `python -c "from config import settings; print(settings.VIDEO_SOURCE)"`

---

## Phase 2A: CV 파이프라인 (cv)

> Phase 1 완료 후 시작. Phase 2B(backend)와 **병렬 진행 가능**.
> DB/웹 의존 없이 독립 동작하도록 구현.

### Step 2A.1 — detector.py (YOLOv8 검출기)

- **담당**: `cv`
- **선행 조건**: Phase 1 완료
- **작업 내용**:
  - `PersonDetector` 클래스 구현
  - `__init__`: YOLO 모델 로드 (`ultralytics.YOLO`), CUDA 디바이스 설정, confidence threshold 적용
  - `detect(frame)` → `sv.Detections` 반환: person 클래스만 필터링 (`class_id == 0`)
  - ROI 영역 크롭 후 검출 지원 (`config.settings`의 ROI 좌표 사용)
- **✅ 검증**: 테스트 이미지 1장으로 바운딩박스 출력 확인

### Step 2A.2 — tracker.py (ByteTrack 추적기)

- **담당**: `cv`
- **선행 조건**: Step 2A.1
- **작업 내용**:
  - `PersonTracker` 클래스 구현
  - `__init__`: `sv.ByteTrack()` 초기화
  - `update(detections)` → tracked `sv.Detections` (ID 포함) 반환
  - `predict()` → 칼만 필터 예측만 수행 (Frame Skip 프레임용)
  - Frame Skip 로직: `frame_count % FRAME_SKIP == 0`일 때만 `detect()` 호출, 나머지는 `predict()`
- **✅ 검증**: 짧은 영상(5초)으로 ID 할당 및 유지 확인, ID가 10+ 프레임 이상 안정적 추적

### Step 2A.3 — counter.py (라인 크로싱 카운터)

- **담당**: `cv`
- **선행 조건**: Step 2A.2
- **작업 내용**:
  - `LineCrossCounter` 클래스 구현
  - `__init__`: 기준선 좌표 (`LINE_START_X/Y`, `LINE_END_X/Y`), 방향 설정
  - `update(tracked_detections)` → `List[CountEvent]` 반환
    - `CountEvent`: `person_id`, `direction` (IN/OUT), `count_change` (+1/-1), `confidence`, `bbox`
  - 중심좌표 계산: `cx = (x1+x2)/2`, `cy = (y1+y2)/2`
  - 라인 통과 판정: 이전 프레임 y값과 현재 y값 비교 (대각선은 직선 방정식 기준 부호 변화)
  - 중복 방지: `counted_ids: Set[int]`, ID별 마지막 통과 방향 기록 `last_direction: Dict[int, str]`
  - `current_count` 속성 (현재 총 인원수)
- **✅ 검증**: 기준선 위→아래 이동 시 OUT, 아래→위 시 IN 정확히 카운트. 동일 ID 중복 카운트 없음

### Step 2A.4 — snapshot.py (스냅샷 관리)

- **담당**: `cv`
- **선행 조건**: Step 2A.3
- **작업 내용**:
  - `SnapshotManager` 클래스 구현
  - `save(frame, event: CountEvent)` → 파일 경로(`str`) 반환
  - 저장 경로: `snapshots/{camera_id}/{YYYYMMDD}/{person_id}_{timestamp}.jpg`
  - OpenCV `cv2.imwrite()` 사용, JPEG 품질 85
  - 바운딩박스 영역 크롭 저장 (전체 프레임 + 크롭 둘 다)
- **✅ 검증**: 이벤트 발생 시 `snapshots/` 폴더에 이미지 파일 생성 확인

### Step 2A.5 — monitor.py (성능 모니터링)

- **담당**: `cv`
- **선행 조건**: Step 2A.1 (2A.2~2A.4와 **병렬 가능**)
- **작업 내용**:
  - `PerformanceMonitor` 클래스 구현
  - FPS 측정: 프레임 처리 시간 역수 (이동 평균)
  - YOLO 추론 시간 측정 (ms)
  - ByteTrack 업데이트 시간 측정 (ms)
  - GPU 메모리 사용량: `pynvml` → `nvmlDeviceGetMemoryInfo()`
  - `get_stats()` → `dict` (fps, inference_ms, tracking_ms, gpu_memory_used, gpu_memory_total)
- **✅ 검증**: 10프레임 처리 후 `get_stats()` 리턴값 정상 확인

### Step 2A.6 — 카운팅 라인 설정 UI

- **담당**: `cv`
- **선행 조건**: Step 2A.3
- **작업 내용**:
  - `LineConfigurator` 클래스 (별도 `line_config.py` 또는 `counter.py` 내)
  - OpenCV `cv2.setMouseCallback()` 기반 마우스 드래그로 라인 설정
  - 미리보기: 현재 프레임에 라인 오버레이 표시
  - 수평선/대각선 모두 지원
  - 확정 시 `.env` 파일에 `LINE_START_X/Y`, `LINE_END_X/Y` 업데이트
- **✅ 검증**: 마우스 드래그로 라인 설정 → `.env` 값 변경 확인

---

## Phase 2B: 백엔드 (backend)

> Phase 1 완료 후 시작. Phase 2A(cv)와 **병렬 진행 가능**.

### Step 2B.1 — database.py (DB 모델 및 연결)

- **담당**: `backend`
- **선행 조건**: Phase 1 완료
- **작업 내용**:
  - SQLAlchemy 엔진 생성 (`create_engine` + PyMySQL)
  - `Base = declarative_base()`
  - `TrackingEvent` 모델 (기획서 §5.1 테이블 스키마 그대로)
  - `SessionLocal` 팩토리 (`sessionmaker`)
  - `get_db()` 의존성 주입 함수 (FastAPI용)
  - DB 초기화 함수: `init_db()` → `Base.metadata.create_all()`
  - CRUD 함수:
    - `save_event(event_data)` → 이벤트 1건 저장
    - `get_current_count(camera_id)` → 최신 `current_count` 조회
    - `get_events(camera_id, start_time, end_time)` → 시간 범위 이벤트 조회
- **✅ 검증**: MySQL 연결 테스트 + 이벤트 1건 INSERT/SELECT 성공

### Step 2B.2 — alert.py (카카오톡 알림)

- **담당**: `backend`
- **선행 조건**: Phase 1 완료 (Step 2B.1과 **병렬 가능**)
- **작업 내용**:
  - `KakaoAlert` 클래스 구현
  - `__init__`: access_token, refresh_token, threshold, cooldown 설정
  - `check_and_send(current_count)`: 임계값 초과 시 알림 발송
  - 카카오 REST API ("나에게 보내기") 호출: `POST https://kapi.kakao.com/v2/api/talk/memo/default/send`
  - 쿨다운 로직: `last_alert_time` 기록, `ALERT_COOLDOWN_SEC` 간격 미만이면 스킵
  - 토큰 만료 시 자동 갱신: refresh_token → 새 access_token 발급
- **✅ 검증**: 테스트 메시지 전송 성공 (카카오 디벨로퍼 앱 등록 필요)

### Step 2B.3 — dashboard/app.py (FastAPI 서버)

- **담당**: `backend`
- **선행 조건**: Step 2B.1
- **작업 내용**:
  - FastAPI 앱 인스턴스 생성
  - **REST API 엔드포인트**:
    - `GET /api/status` → 현재 인원수, 누적 IN/OUT
    - `GET /api/events?camera_id=&start=&end=` → 이벤트 목록 조회
    - `GET /api/stats` → 성능 모니터링 데이터 (FPS, GPU 등)
  - **WebSocket 엔드포인트**:
    - `WS /ws/counter` → 실시간 카운트 변화 push
    - `WS /ws/stream` → 실시간 영상 프레임 push (옵션, JPEG 바이너리)
  - Jinja2 템플릿 렌더링 (`GET /` → 대시보드 HTML)
  - CORS 미들웨어 설정
  - startup 이벤트: DB 초기화 (`init_db()`)
- **✅ 검증**: `uvicorn dashboard.app:app --reload` 실행 후 `/api/status` 200 응답 확인

### Step 2B.4 — 상태 복구 로직

- **담당**: `backend`
- **선행 조건**: Step 2B.1
- **작업 내용**:
  - `StateRecovery` 클래스 (database.py 내 또는 별도 파일)
  - `restore_count(camera_id)` → DB에서 해당 카메라 마지막 `current_count` 조회
  - `save_state(camera_id, counted_ids, current_count)` → 주기적 상태 저장 (30초 간격)
  - main.py 시작 시 `restore_count()` 호출하여 카운터 초기값 설정
- **✅ 검증**: 카운트 10까지 올린 후 프로세스 종료 → 재시작 시 카운트 10에서 복원

---

## Phase 3: 프론트엔드 (frontend)

> Step 2B.3 완료 후 시작. FastAPI 서버/API가 동작해야 연동 가능.

### Step 3.1 — 대시보드 메인 페이지 (index.html)

- **담당**: `frontend`
- **선행 조건**: Step 2B.3
- **작업 내용**:
  - `dashboard/templates/index.html` 작성
  - 레이아웃: 상단 헤더 + 카드 그리드 + 하단 테이블
  - **카드 영역**:
    - 현재 인원수 (큰 숫자, 실시간 업데이트)
    - 누적 IN 카운트
    - 누적 OUT 카운트
    - 카메라 ID 표시
  - CSS: 반응형 디자인 (모바일/태블릿/데스크탑)
- **✅ 검증**: 브라우저에서 `http://localhost:8000/` 접속 시 레이아웃 정상 렌더링

### Step 3.2 — WebSocket 실시간 연동

- **담당**: `frontend`
- **선행 조건**: Step 3.1
- **작업 내용**:
  - JavaScript WebSocket 클라이언트 구현
  - `ws://localhost:8000/ws/counter` 연결
  - 메시지 수신 시 카드 숫자 실시간 업데이트 (DOM 조작)
  - 연결 끊김 시 자동 재연결 (exponential backoff)
  - 연결 상태 표시 (🟢 연결됨 / 🔴 끊김)
- **✅ 검증**: 카운트 이벤트 발생 시 페이지 새로고침 없이 숫자 변동 확인

### Step 3.3 — 차트/그래프

- **담당**: `frontend`
- **선행 조건**: Step 3.1 (Step 3.2와 **병렬 가능**)
- **작업 내용**:
  - Chart.js CDN 로드
  - 시간대별 인원 추이 라인 차트 (최근 1시간, 5분 단위)
  - IN/OUT 비율 도넛 차트
  - `/api/events` 호출하여 데이터 로드
  - 주기적 자동 갱신 (30초 간격)
- **✅ 검증**: 이벤트 데이터 10건 이상일 때 차트 정상 렌더링

### Step 3.4 — 성능 모니터링 패널

- **담당**: `frontend`
- **선행 조건**: Step 3.2
- **작업 내용**:
  - `/api/stats` 호출하여 성능 데이터 표시
  - FPS 게이지, 추론 시간, 추적 시간, GPU 메모리 사용량
  - 5초 간격 자동 갱신
- **✅ 검증**: GPU 모니터링 값이 실시간으로 변동 확인

### Step 3.5 — 실시간 영상 스트리밍 (옵션)

- **담당**: `frontend`
- **선행 조건**: Step 3.2
- **작업 내용**:
  - `WS /ws/stream`에서 MJPEG 프레임 수신
  - `<canvas>` 또는 `<img>` 태그에 프레임 표시
  - 바운딩박스 + 카운팅 라인 오버레이 표시
  - FPS에 따른 프레임 드롭 처리
- **✅ 검증**: 대시보드에서 실시간 영상 + 오버레이 확인

---

## Phase 4: 통합 (cv + backend)

> Phase 2A + 2B + 3 모두 완료 후 시작.

### Step 4.1 — main.py 통합 진입점

- **담당**: `cv` + `backend`
- **선행 조건**: Phase 2A, 2B, 3 완료
- **작업 내용**:
  - 모든 모듈 import 및 조합
  - 실행 흐름:
    1. `config.settings` 로드
    2. `StateRecovery.restore_count()` → 카운터 초기값 복원
    3. `PersonDetector` 초기화
    4. `PersonTracker` 초기화
    5. `LineCrossCounter` 초기화 (복원된 카운트로)
    6. `SnapshotManager` 초기화
    7. `PerformanceMonitor` 초기화
    8. `KakaoAlert` 초기화
    9. FastAPI 서버 별도 스레드로 실행 (`uvicorn` in-process)
    10. 메인 루프: VideoCapture → detect/track → count → 이벤트 처리 (DB 저장, 알림, 스냅샷, WebSocket push)
  - 종료 처리: `Ctrl+C` → 상태 저장 → 리소스 해제
- **✅ 검증**: `python main.py` 실행 → 영상에서 사람 감지 + 카운팅 + 대시보드 동시 동작

### Step 4.2 — 모듈 간 이벤트 연결

- **담당**: `backend`
- **선행 조건**: Step 4.1
- **작업 내용**:
  - CV 파이프라인 → DB 저장: `CountEvent` → `database.save_event()`
  - CV 파이프라인 → 알림: `CountEvent` → `alert.check_and_send()`
  - CV 파이프라인 → 스냅샷: `CountEvent` → `snapshot.save()`
  - CV 파이프라인 → WebSocket: `CountEvent` → 대시보드 실시간 push
  - 성능 데이터 → WebSocket/API: `monitor.get_stats()` → `/api/stats`
- **✅ 검증**: 1명 라인 통과 시 DB 기록 + 스냅샷 저장 + 대시보드 업데이트 모두 동시 발생

---

## Phase 5: 테스트 (tester)

> Phase 4 완료 후 시작.

### Step 5.1 — 단위 테스트 작성

- **담당**: `tester`
- **선행 조건**: Phase 4 완료
- **작업 내용**:
  - `tests/test_detector.py`: 모델 로드, person 필터링, confidence 임계값
  - `tests/test_tracker.py`: ID 할당, ID 유지, 칼만 필터 예측
  - `tests/test_counter.py`: 라인 통과 IN/OUT 판정, 중복 방지, 대각선 라인
  - `tests/test_database.py`: CRUD 함수, 연결 실패 처리
  - `tests/test_alert.py`: 쿨다운 로직, 토큰 갱신 (mock)
  - `tests/test_snapshot.py`: 파일 저장, 경로 생성
- **✅ 검증**: `pytest tests/ -v` 전체 통과

### Step 5.2 — 통합 테스트

- **담당**: `tester`
- **선행 조건**: Step 5.1
- **작업 내용**:
  - `tests/test_integration.py`: 전체 파이프라인 end-to-end (테스트 영상 사용)
  - `tests/test_api.py`: FastAPI TestClient로 REST API/WebSocket 테스트
  - `tests/test_state_recovery.py`: 상태 저장/복구 시나리오
  - 테스트 픽스처: 짧은 테스트 영상 파일 (`tests/fixtures/test_video.mp4`)
- **✅ 검증**: `pytest tests/ -v --tb=short` 전체 통과, 커버리지 80% 이상

### Step 5.3 — 엣지 케이스 검증

- **담당**: `tester`
- **선행 조건**: Step 5.2
- **작업 내용**:
  - 사람 0명 영상 → 카운트 0 유지
  - 다수(10명+) 동시 통과 → 모두 정확히 카운트
  - 빠른 이동 (프레임 사이 큰 위치 변화) → ID 유지 여부
  - RTSP 연결 끊김 → 재연결 처리
  - DB 연결 실패 → 로그 기록 후 계속 동작 (graceful degradation)
- **✅ 검증**: 각 엣지 케이스별 테스트 시나리오 통과

---

## Phase 6: 배포 준비 (devops)

> Phase 5 완료 후 시작.

### Step 6.1 — Docker 컨테이너화

- **담당**: `devops`
- **선행 조건**: Phase 5 완료
- **작업 내용**:
  - `Dockerfile` 작성: Python 3.12 + CUDA 베이스 이미지, requirements.txt 설치, 앱 복사
  - `docker-compose.yml`: app 서비스 + MySQL 서비스 + 볼륨 (snapshots, models)
  - GPU 패스스루 설정 (`deploy.resources.reservations.devices`)
- **✅ 검증**: `docker-compose up` → 전체 시스템 정상 동작

### Step 6.2 — 운영 설정

- **담당**: `devops`
- **선행 조건**: Step 6.1
- **작업 내용**:
  - 로그 설정: loguru 파일 로테이션 (`10MB`, 7일 보관)
  - systemd 서비스 파일 또는 supervisor 설정 (Linux 배포 시)
  - 스냅샷 디렉터리 자동 정리 (30일 초과 파일 삭제 cron)
- **✅ 검증**: 24시간 장기 운영 테스트 → 메모리 누수/디스크 풀 없음

---

## 워크플로우 의존성 다이어그램

```
Phase 1 (devops: 초기화)
    │
    ├──→ Phase 2A (cv: 파이프라인)  ──┐
    │    2A.1 → 2A.2 → 2A.3          │
    │    → 2A.4, 2A.5(병렬), 2A.6    │
    │                                  ├──→ Phase 4 (통합)
    └──→ Phase 2B (backend: 서버)  ───┘        │
         2B.1, 2B.2(병렬)                       │
         → 2B.3, 2B.4                           ▼
              │                           Phase 5 (tester: 검증)
              └──→ Phase 3 (frontend)            │
                   3.1 → 3.2, 3.3(병렬)         ▼
                   → 3.4, 3.5             Phase 6 (devops: 배포)
```

---

## 에이전트별 작업 요약

| 에이전트     | 담당 Step                               | 총 작업 수 |
| ------------ | --------------------------------------- | :--------: |
| **devops**   | 1.1, 1.2, 1.3, 1.4, 6.1, 6.2            |     6      |
| **cv**       | 2A.1, 2A.2, 2A.3, 2A.4, 2A.5, 2A.6, 4.1 |     7      |
| **backend**  | 2B.1, 2B.2, 2B.3, 2B.4, 4.2             |     5      |
| **frontend** | 3.1, 3.2, 3.3, 3.4, 3.5                 |     5      |
| **tester**   | 5.1, 5.2, 5.3                           |     3      |
| **합계**     |                                         |   **26**   |
