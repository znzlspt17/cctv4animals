from server.repositories.base import AbstractRepository
from server.repositories.mysql_repo import PostgresRepository


def get_repository() -> AbstractRepository:
    return PostgresRepository()
