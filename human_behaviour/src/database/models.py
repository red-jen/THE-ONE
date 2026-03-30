"""SQLAlchemy ORM models — PostgreSQL schema for Protest Leader Detection."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, String, Text, Boolean, DateTime,
    ForeignKey, JSON, Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String(255), nullable=False, index=True)
    status = Column(String(50), default="pending")
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    videos_count = Column(Integer, default=0)
    scoring_backend = Column(String(50), default="rag_llm")
    manifest = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    videos = relationship("Video", back_populates="pipeline_run", cascade="all, delete-orphan")
    persons = relationship("Person", back_populates="pipeline_run", cascade="all, delete-orphan")
    queries = relationship("Query", back_populates="pipeline_run", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    camera_id = Column(String(255), nullable=False)
    filename = Column(String(500), nullable=False)
    filepath = Column(Text, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=_utcnow)

    pipeline_run = relationship("PipelineRun", back_populates="videos")


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    person_id = Column(Integer, nullable=False)
    observations = Column(Integer, default=0)
    cameras_seen = Column(Integer, default=0)
    duration_sec = Column(Float, default=0.0)
    front_count = Column(Integer, default=0)
    center_count = Column(Integer, default=0)
    back_count = Column(Integer, default=0)
    megaphone_count = Column(Integer, default=0)
    banner_count = Column(Integer, default=0)
    flag_count = Column(Integer, default=0)
    microphone_count = Column(Integer, default=0)
    gesture_total = Column(Integer, default=0)

    leader_score = Column(Float, nullable=True)
    heuristic_score = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    scoring_backend = Column(String(50), nullable=True)

    pipeline_run = relationship("PipelineRun", back_populates="persons")
    descriptions = relationship("Description", back_populates="person", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_persons_run_person", "run_id", "person_id", unique=True),
    )


class Description(Base):
    __tablename__ = "descriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_db_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    frame_idx = Column(Integer, nullable=False)
    timestamp_sec = Column(Float, default=0.0)
    camera_id = Column(String(255), nullable=True)
    description_text = Column(Text, nullable=True)
    bbox = Column(JSON, nullable=True)
    detection_confidence = Column(Float, nullable=True)
    signals = Column(JSON, nullable=True)

    person = relationship("Person", back_populates="descriptions")


class LeaderScore(Base):
    """Stores the final RAG-scored leadership ranking per run."""
    __tablename__ = "leader_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    person_id = Column(Integer, nullable=False)
    leader_score = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    backend_used = Column(String(50), nullable=True)
    scored_at = Column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("ix_leader_scores_run", "run_id"),
    )


class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    generator_backend = Column(String(50), nullable=True)
    model_used = Column(String(100), nullable=True)
    evidence_count = Column(Integer, default=0)
    asked_at = Column(DateTime(timezone=True), default=_utcnow)

    pipeline_run = relationship("PipelineRun", back_populates="queries")
