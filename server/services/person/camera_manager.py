"""다중 카메라를 통합 관리하는 CameraManager.

사용 예::

    manager = CameraManager()
    manager.start_all(configs, repo)   # 전체 시작 (lifespan)
    manager.stop_all()                 # 전체 종료 (lifespan)

    # 런타임 카메라 추가 / 제거
    manager.add_camera(new_cfg, repo)
    manager.remove_camera("cam_03")
"""

import logging
import threading
from typing import Any, Iterator

from server.services.person.camera_config import CameraConfig
from server.services.person.person_service import PersonCounterService

logger = logging.getLogger(__name__)


class CameraManager:
    """여러 대의 카메라(PersonCounterService)를 생명주기 단위로 관리한다."""

    def __init__(self) -> None:
        self._services: dict[str, PersonCounterService] = {}
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────
    # 전체 생명주기
    # ──────────────────────────────────────────────

    def start_all(self, configs: list[CameraConfig], repo: Any) -> None:
        """설정 목록의 모든 카메라를 순서대로 시작한다."""
        for cfg in configs:
            self.add_camera(cfg, repo)

    def stop_all(self) -> None:
        """실행 중인 모든 카메라를 안전하게 종료한다."""
        with self._lock:
            camera_ids = list(self._services.keys())
        for camera_id in camera_ids:
            self._stop_one(camera_id)

    # ──────────────────────────────────────────────
    # 개별 카메라 관리
    # ──────────────────────────────────────────────

    def add_camera(self, config: CameraConfig, repo: Any) -> PersonCounterService:
        """카메라를 추가하고 스레드를 시작한다.

        동일한 camera_id 가 이미 실행 중이면 기존 서비스를 반환한다.
        """
        with self._lock:
            if config.camera_id in self._services:
                logger.warning(
                    "CameraManager: camera_id=%s already running, skipping",
                    config.camera_id,
                )
                return self._services[config.camera_id]

            svc = PersonCounterService(config)
            self._services[config.camera_id] = svc

        svc.start(repo)
        logger.info("CameraManager: added camera %s", config.camera_id)
        return svc

    def remove_camera(self, camera_id: str) -> bool:
        """카메라를 중지하고 목록에서 제거한다.

        Returns:
            ``True`` — 정상 제거, ``False`` — 존재하지 않는 ID.
        """
        return self._stop_one(camera_id, remove=True)

    def _stop_one(self, camera_id: str, *, remove: bool = True) -> bool:
        with self._lock:
            svc = self._services.get(camera_id)
            if svc is None:
                logger.warning("CameraManager: camera_id=%s not found", camera_id)
                return False

        svc.stop()
        if remove:
            with self._lock:
                self._services.pop(camera_id, None)
            logger.info("CameraManager: removed camera %s", camera_id)
        return True

    # ──────────────────────────────────────────────
    # 조회
    # ──────────────────────────────────────────────

    def get(self, camera_id: str) -> PersonCounterService | None:
        """camera_id 로 서비스 인스턴스를 반환한다."""
        with self._lock:
            return self._services.get(camera_id)

    def __iter__(self) -> Iterator[PersonCounterService]:
        with self._lock:
            return iter(list(self._services.values()))

    def list_stats(self) -> list[dict[str, Any]]:
        """모든 카메라의 상태 딕셔너리 목록을 반환한다."""
        with self._lock:
            services = list(self._services.values())
        return [svc.get_stats() for svc in services]

    def aggregate_count(self) -> dict[str, Any]:
        """전체 카메라의 합산 카운트를 반환한다."""
        with self._lock:
            services = list(self._services.values())
        total_current = sum(s.current_count for s in services)
        total_in = sum(s.in_count for s in services)
        total_out = sum(s.out_count for s in services)
        return {
            "total_current_count": total_current,
            "total_in_count": total_in,
            "total_out_count": total_out,
            "camera_count": len(services),
        }

    # ──────────────────────────────────────────────
    # 카메라별 pause / resume
    # ──────────────────────────────────────────────

    def pause_camera(self, camera_id: str, timeout: float = 5.0) -> bool:
        svc = self.get(camera_id)
        if svc is None:
            raise KeyError(f"camera_id={camera_id!r} not found")
        return svc.pause_camera(timeout=timeout)

    def resume_camera(self, camera_id: str) -> None:
        svc = self.get(camera_id)
        if svc is None:
            raise KeyError(f"camera_id={camera_id!r} not found")
        svc.resume_camera()
