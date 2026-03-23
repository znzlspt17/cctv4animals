from server.config import settings
from server.repositories.base import AbstractRepository


def get_repository() -> AbstractRepository:
    if settings.DB_BACKEND == "redis":
        from server.repositories.redis_repo import RedisRepository

        return RedisRepository()
    else:
        from server.repositories.mysql_repo import MySQLRepository

        return MySQLRepository()
