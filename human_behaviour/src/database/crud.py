"""CRUD operations for the protest leader detection database."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from .models import PipelineRun, Video, Person, Description, LeaderScore, Query


# ---------------------------------------------------------------------------
# Pipeline runs
# ---------------------------------------------------------------------------

def create_run(db: Session, run_name: str, videos_count: int = 0) -> PipelineRun:
    run = PipelineRun(run_name=run_name, status="running", videos_count=videos_count)
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def complete_run(
    db: Session, run_id: int,
    status: str = "completed",
    manifest: dict | None = None,
    error_message: str | None = None,
    scoring_backend: str | None = None,
) -> PipelineRun:
    run = db.query(PipelineRun).filter(PipelineRun.id == run_id).first()
    if run:
        run.status = status
        run.completed_at = datetime.now(timezone.utc)
        if manifest:
            run.manifest = manifest
        if error_message:
            run.error_message = error_message
        if scoring_backend:
            run.scoring_backend = scoring_backend
        db.commit()
        db.refresh(run)
    return run


def get_run(db: Session, run_id: int) -> Optional[PipelineRun]:
    return db.query(PipelineRun).filter(PipelineRun.id == run_id).first()


def get_run_by_name(db: Session, run_name: str) -> Optional[PipelineRun]:
    return (
        db.query(PipelineRun)
        .filter(PipelineRun.run_name == run_name)
        .order_by(PipelineRun.created_at.desc())
        .first()
    )


def list_runs(db: Session, limit: int = 50, offset: int = 0) -> List[PipelineRun]:
    return (
        db.query(PipelineRun)
        .order_by(PipelineRun.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

def add_video(
    db: Session, run_id: int,
    camera_id: str, filename: str, filepath: str,
) -> Video:
    v = Video(run_id=run_id, camera_id=camera_id, filename=filename, filepath=filepath)
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


# ---------------------------------------------------------------------------
# Persons + descriptions
# ---------------------------------------------------------------------------

def upsert_person(
    db: Session, run_id: int, person_id: int, **kwargs,
) -> Person:
    p = (
        db.query(Person)
        .filter(Person.run_id == run_id, Person.person_id == person_id)
        .first()
    )
    if p is None:
        p = Person(run_id=run_id, person_id=person_id, **kwargs)
        db.add(p)
    else:
        for k, v in kwargs.items():
            setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return p


def add_description(db: Session, person_db_id: int, **kwargs) -> Description:
    d = Description(person_db_id=person_db_id, **kwargs)
    db.add(d)
    db.commit()
    return d


def get_persons_for_run(db: Session, run_id: int) -> List[Person]:
    return (
        db.query(Person)
        .filter(Person.run_id == run_id)
        .order_by(Person.leader_score.desc().nullslast())
        .all()
    )


# ---------------------------------------------------------------------------
# Leader scores
# ---------------------------------------------------------------------------

def save_leader_scores(
    db: Session, run_id: int,
    candidates: List[dict],
    backend_used: str,
) -> List[LeaderScore]:
    saved = []
    for c in candidates:
        ls = LeaderScore(
            run_id=run_id,
            person_id=c["person_id"],
            leader_score=float(c.get("leader_score", 0)),
            reasoning=c.get("reasoning"),
            backend_used=backend_used,
        )
        db.add(ls)
        saved.append(ls)

        upsert_person(
            db, run_id=run_id, person_id=c["person_id"],
            leader_score=float(c.get("leader_score", 0)),
            reasoning=c.get("reasoning"),
            scoring_backend=backend_used,
        )

    db.commit()
    return saved


def get_leader_scores(db: Session, run_id: int) -> List[LeaderScore]:
    return (
        db.query(LeaderScore)
        .filter(LeaderScore.run_id == run_id)
        .order_by(LeaderScore.leader_score.desc())
        .all()
    )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def save_query(
    db: Session, run_id: int | None,
    question: str, answer: str,
    generator_backend: str, model_used: str | None,
    evidence_count: int = 0,
) -> Query:
    q = Query(
        run_id=run_id,
        question=question,
        answer=answer,
        generator_backend=generator_backend,
        model_used=model_used,
        evidence_count=evidence_count,
    )
    db.add(q)
    db.commit()
    db.refresh(q)
    return q


def get_queries(db: Session, run_id: int, limit: int = 20) -> List[Query]:
    return (
        db.query(Query)
        .filter(Query.run_id == run_id)
        .order_by(Query.asked_at.desc())
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# Bulk import from JSONL files into DB
# ---------------------------------------------------------------------------

def import_descriptions_from_jsonl(
    db: Session, run_id: int, jsonl_path: str,
) -> int:
    """Import description JSONL (pipeline output) into the database."""
    rows = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    person_cache = {}
    for row in rows:
        pid = int(row["person_id"])
        if pid not in person_cache:
            p = upsert_person(db, run_id=run_id, person_id=pid)
            person_cache[pid] = p

        add_description(
            db,
            person_db_id=person_cache[pid].id,
            frame_idx=int(row.get("frame_idx", 0)),
            timestamp_sec=float(row.get("timestamp_sec", 0)),
            camera_id=str(row.get("camera_id", "")),
            description_text=row.get("description", ""),
            bbox=row.get("bbox"),
            detection_confidence=float(row.get("detection_confidence", 0)),
            signals=row.get("signals"),
        )

    return len(rows)
