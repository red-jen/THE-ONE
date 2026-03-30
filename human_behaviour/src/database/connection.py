"""Database connection management via SQLAlchemy.

Supports PostgreSQL (production) and SQLite (local dev fallback).
Set DATABASE_URL (environment or human_behaviour/.env) to switch; restart the
process after changing it. Shell environment wins over .env when both are set.
"""

from __future__ import annotations

import os
from pathlib import Path

import env_bootstrap  # noqa: F401 — loads project .env before getenv below
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_SQLITE = "sqlite:///" + str(
    (_ROOT / "data" / "protest_leader.db").as_posix()
)


def _normalize_database_url(raw: str | None) -> str:
    """Resolve URL from env; default SQLite; ensure psycopg2 driver for Postgres."""
    url = (raw or "").strip()
    if not url:
        return _DEFAULT_SQLITE
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and not url.startswith("postgresql+"):
        return "postgresql+psycopg2://" + url[len("postgresql://") :]
    return url


DATABASE_URL = _normalize_database_url(os.getenv("DATABASE_URL"))

_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args=_connect_args,
    **({"pool_size": 5, "max_overflow": 10} if not DATABASE_URL.startswith("sqlite") else {}),
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once at startup."""
    from .models import Base
    if DATABASE_URL.startswith("sqlite"):
        db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Initialized: {DATABASE_URL}")
