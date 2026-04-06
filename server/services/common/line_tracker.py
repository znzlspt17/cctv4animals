"""범용 라인 크로싱 추적기.

부호 거리(signed distance) 방식으로 객체가 가상의 선을 교차했는지 감지한다.
person_counter 와 animal detection 양쪽에서 공유하는 공통 로직.
"""

import logging

logger = logging.getLogger(__name__)


class LineCrossTracker:
    """임의의 직선을 기준으로 객체의 라인 교차를 추적한다.

    직선 방정식: a*x + b*y + c = 0
    부호 거리가 음→양, 또는 양→음으로 반전되면 라인을 교차한 것으로 판단.

    Args:
        x1, y1: 라인 시작점 (픽셀)
        x2, y2: 라인 끝점 (픽셀)
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self._a = y2 - y1
        self._b = x1 - x2
        self._c = x2 * y1 - x1 * y2
        # track_id → 이전 프레임 부호 거리
        self._prev_dist: dict[int, float] = {}
        logger.debug(
            "LineCrossTracker initialized: line=(%s,%s)→(%s,%s)", x1, y1, x2, y2
        )

    def signed_distance(self, cx: float, cy: float) -> float:
        """점 (cx, cy)의 직선에 대한 부호 거리를 반환한다."""
        return self._a * cx + self._b * cy + self._c

    def update(
        self,
        centers: list[tuple[float, float]],
        track_ids: list[int],
    ) -> list[tuple[int, str]]:
        """매 프레임마다 호출한다.

        Args:
            centers: 각 객체의 중심 좌표 목록 [(cx, cy), ...]
            track_ids: centers 와 1:1 대응하는 추적 ID 목록

        Returns:
            라인을 교차한 객체의 (track_id, direction) 목록.
            direction 은 ``"IN"`` 또는 ``"OUT"``.
            교차가 없으면 빈 리스트.
        """
        events: list[tuple[int, str]] = []

        for (cx, cy), track_id in zip(centers, track_ids):
            curr_dist = self.signed_distance(cx, cy)

            if track_id in self._prev_dist:
                prev_dist = self._prev_dist[track_id]
                if prev_dist * curr_dist < 0:  # 부호 반전 = 라인 교차
                    direction = "IN" if prev_dist < 0 else "OUT"
                    events.append((track_id, direction))
                    logger.debug(
                        "LineCross: track_id=%d direction=%s", track_id, direction
                    )

            self._prev_dist[track_id] = curr_dist

        # 사라진 track_id 정리
        active = set(track_ids)
        for tid in [t for t in list(self._prev_dist) if t not in active]:
            del self._prev_dist[tid]

        return events

    def reset(self) -> None:
        """추적 상태를 초기화한다 (카메라 전환 등)."""
        self._prev_dist.clear()
