"""단일 카메라 설정을 담는 데이터 클래스.

CameraManager 가 카메라 목록을 관리할 때 각 항목으로 사용한다.
환경 변수 ``CAMERAS_JSON`` 에 JSON 배열로 여러 대를 정의하거나,
하위 호환을 위해 기존 ``PERSON_*`` 단일 설정도 그대로 지원한다.

JSON 예시 (CAMERAS_JSON)::

    [
      {
        "camera_id": "cam_01",
        "video_source": "0",
        "label": "정문",
        "line_start_x": 0,
        "line_start_y": 360,
        "line_end_x": 1280,
        "line_end_y": 360
      },
      {
        "camera_id": "cam_02",
        "video_source": "rtsp://192.168.1.10/stream",
        "label": "후문",
        "line_start_x": 0,
        "line_start_y": 540,
        "line_end_x": 1920,
        "line_end_y": 540
      }
    ]
"""

from dataclasses import dataclass


@dataclass
class CameraConfig:
    """단일 카메라에 대한 런타임 설정."""

    # 필수
    camera_id: str
    """고유 카메라 식별자 (DB camera_id 컬럼과 일치해야 함)."""
    video_source: str
    """cv2.VideoCapture 에 전달할 소스.
    숫자 문자열(``"0"``, ``"1"``) → 정수로 변환, 나머지는 그대로 사용.
    RTSP URL, 파일 경로 모두 가능."""

    # 선택 — 라인 크로싱
    line_start_x: int = 0
    line_start_y: int = 360
    line_end_x: int = 1280
    line_end_y: int = 360

    # 선택 — 처리 옵션
    frame_skip: int = 10
    """N 프레임마다 1회 추론 (낮을수록 CPU 부하 증가)."""
    confidence_threshold: float = 0.5

    # 선택 — ROI (0 이면 전체 프레임 사용)
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 0
    roi_h: int = 0

    # 선택 — 메타
    label: str = ""
    """사람이 읽기 쉬운 카메라 이름 (예: '정문', '2층 복도')."""
    snapshot_dir: str = "snapshots"

    @property
    def capture_source(self) -> int | str:
        """cv2.VideoCapture 에 넘길 실제 소스 값을 반환한다."""
        if self.video_source.isdigit():
            return int(self.video_source)
        return self.video_source
