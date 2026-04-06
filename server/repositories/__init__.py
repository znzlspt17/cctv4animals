from server.repositories.base import AbstractRepository
from server.repositories.postgres_repo import PostgresRepository


def get_repository() -> AbstractRepository:
    return PostgresRepository()
