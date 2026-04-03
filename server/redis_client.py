# Redis는 제거되었습니다. 원격 PostgreSQL을 사용합니다.
# 이 파일은 하위 호환성을 위해 유지되나 사용되지 않습니다.


def get_redis_client():
    """Redis 제거 후 스텁 — 실제 연결을 시도하지 않습니다."""
    raise NotImplementedError("Redis is not available. Use PostgreSQL instead.")


def redis_client():
    return get_redis_client()
