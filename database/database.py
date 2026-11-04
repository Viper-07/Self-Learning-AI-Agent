"""
database/database.py — Async SQLAlchemy engine, session factory, and lifecycle.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from sqlite3 import Connection as SQLite3Connection
from typing import AsyncIterator

import structlog
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import get_settings
from database.models import Base

logger = structlog.get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _session_factory


async def init_db(settings=None) -> None:
    """Create engine, ensure data directory exists, and run DDL migrations.

    Must be called once at application startup before any DB access.
    """
    global _engine, _session_factory
    if settings is None:
        settings = get_settings()

    is_sqlite = "sqlite" in settings.database_url
    connect_args = {"check_same_thread": False} if is_sqlite else {}

    # If SQLite file path, ensure the data directory exists
    if is_sqlite and ":memory:" not in settings.database_url:
        db_path = settings.database_url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _engine = create_async_engine(
        settings.database_url,
        echo=settings.api_debug,
        connect_args=connect_args,
    )

    if is_sqlite:
        @event.listens_for(_engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            if isinstance(dbapi_connection, SQLite3Connection):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON;")
                cursor.close()

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("database.ready", url=settings.database_url)


async def close_db() -> None:
    """Dispose the engine. Called on application shutdown."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database.closed")


@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Async context manager yielding a transactional DB session.

    Usage::

        async with get_db_session() as session:
            result = await session.execute(...)

    The session is automatically committed on success and rolled back on error.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a DB session per request."""
    async with get_db_session() as session:
        yield session
