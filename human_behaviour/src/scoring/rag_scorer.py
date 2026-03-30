"""RAG-based leadership scorer — the LLM is the final judge.

Instead of a fixed-weight formula, this module:
1. Aggregates raw signals per person (same data the old scorer used)
2. Retrieves relevant description evidence from the memory store
3. Sends everything to an LLM (via Ollama) which reasons about
   who is a leader, assigns a score 0-100, and explains why
4. Falls back to a deterministic heuristic if the LLM is unavailable
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request


# ---------------------------------------------------------------------------
# Signal aggregation (preprocessing — NOT the decision)
# ---------------------------------------------------------------------------

@dataclass
class PersonEvidence:
    person_id: int
    observations: int = 0
    cameras: set = field(default_factory=set)
    front_count: int = 0
    center_count: int = 0
    back_count: int = 0
    megaphone_count: int = 0
    banner_count: int = 0
    flag_count: int = 0
    microphone_count: int = 0
    gesture_total: int = 0
    confidence_sum: float = 0.0
    min_ts: float = float("inf")
    max_ts: float = 0.0
    descriptions: list = field(default_factory=list)


def aggregate_evidence(rows: List[dict]) -> Dict[int, PersonEvidence]:
    agg: Dict[int, PersonEvidence] = {}

    for row in rows:
        pid = int(row["person_id"])
        signals = row.get("signals", {})
        camera = str(row.get("camera_id", "unknown"))
        ts = float(row.get("timestamp_sec", 0.0))

        p = agg.get(pid)
        if p is None:
            p = PersonEvidence(person_id=pid)
            agg[pid] = p

        p.observations += 1
        p.cameras.add(camera)
        p.confidence_sum += float(row.get("detection_confidence", 0.0))
        p.min_ts = min(p.min_ts, ts)
        p.max_ts = max(p.max_ts, ts)

        pos = str(signals.get("position_hint", "unknown"))
        if pos == "front":
            p.front_count += 1
        elif pos == "center":
            p.center_count += 1
        elif pos == "back":
            p.back_count += 1

        p.megaphone_count += int(bool(signals.get("has_megaphone")))
        p.banner_count += int(bool(signals.get("has_banner")))
        p.flag_count += int(bool(signals.get("has_flag")))
        p.microphone_count += int(bool(signals.get("has_microphone")))
        p.gesture_total += int(signals.get("gesture_score", 0))

        desc_text = row.get("description", "")
        if desc_text:
            p.descriptions.append(
                f"[cam={camera} t={ts:.1f}s] {desc_text}"
            )

    return agg


def person_evidence_to_context(p: PersonEvidence) -> str:
    """Format one person's evidence into a readable text block for the LLM."""
    duration = max(0.0, p.max_ts - (p.min_ts if p.min_ts != float("inf") else 0.0))
    lines = [
        f"Person ID: {p.person_id}",
        f"  Observations: {p.observations}",
        f"  Duration visible: {duration:.1f}s",
        f"  Cameras seen in: {sorted(p.cameras)}",
        f"  Position counts — front: {p.front_count}, center: {p.center_count}, back: {p.back_count}",
        f"  Objects detected — megaphone: {p.megaphone_count}, banner: {p.banner_count}, "
        f"flag: {p.flag_count}, microphone: {p.microphone_count}",
        f"  Gesture score total: {p.gesture_total}",
        f"  Sample descriptions ({min(len(p.descriptions), 5)} of {len(p.descriptions)}):",
    ]
    for d in p.descriptions[:5]:
        lines.append(f"    - {d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-based scoring (the actual judge)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert analyst for protest behavior intelligence.
You are given structured evidence about multiple people detected in protest footage.
Your job is to score each person from 0 to 100 on how likely they are a protest LEADER.

Consider these factors:
- Front-of-crowd positioning (leaders tend to be at the front)
- Possession of communication tools (megaphone, microphone)
- Active gesturing and body language (addressing the crowd)
- Duration of visibility (leaders are persistent, not passing by)
- Presence across multiple camera angles (prominent figures appear everywhere)
- Carrying organizational symbols (banners, flags)
- Overall behavioral pattern described in the evidence text

You MUST respond with valid JSON only — an array of objects:
[
  {
    "person_id": <int>,
    "leader_score": <0-100>,
    "reasoning": "<REQUIRED: 2-4 sentences explaining WHY this person may or may not be a protest leader. Cite concrete evidence: position (front/center/back), megaphone/microphone/banner/flag, gestures, duration, multi-camera presence, and what the descriptions say. Do not only repeat the score.>"
  },
  ...
]

Rank from highest to lowest score. Be decisive — differentiate between people.
If evidence is weak for everyone, scores should be low. Do not inflate.
Every object MUST include a non-empty "reasoning" string."""


def _call_ollama(
    prompt: str,
    model: str = "llama3.1:8b",
    url: str = "http://127.0.0.1:11434",
    timeout_sec: float = 120.0,
    system: str = SYSTEM_PROMPT,
) -> str:
    # Preferred path: LangChain + Ollama.
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOllama(
            model=model,
            base_url=url,
            temperature=0.2,
            num_predict=2048,
            timeout=timeout_sec,
        )
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = str(getattr(resp, "content", "")).strip()
        if text:
            return text
    except Exception:
        # Fallback to direct Ollama HTTP call if LangChain is unavailable.
        pass

    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }
    endpoint = url.rstrip("/") + "/api/generate"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return str(parsed.get("response", "")).strip()


def _parse_llm_scores(text: str) -> List[dict]:
    """Extract the JSON array from LLM output, tolerating markdown fences."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in LLM response: {text[:300]}")
    return json.loads(match.group(0))


def _reasoning_from_evidence(p: PersonEvidence, score: float) -> str:
    """If the LLM omits reasoning, synthesize a short evidence-based explanation."""
    duration = max(0.0, p.max_ts - (p.min_ts if p.min_ts != float("inf") else 0.0))
    bits: List[str] = []
    if p.front_count:
        bits.append(f"toward the front in {p.front_count} observation(s)")
    if p.back_count and p.front_count <= p.back_count:
        bits.append(f"more often toward the back in {p.back_count} observation(s)")
    if p.megaphone_count:
        bits.append(f"megaphone-related cues in {p.megaphone_count} observation(s)")
    if p.microphone_count:
        bits.append("microphone-related cues present")
    if p.banner_count or p.flag_count:
        bits.append(f"banners/flags in {p.banner_count + p.flag_count} observation(s)")
    if p.gesture_total:
        bits.append(f"gesture activity (combined score {p.gesture_total})")
    if len(p.cameras) > 1:
        bits.append(f"seen on {len(p.cameras)} cameras")
    if duration >= 3:
        bits.append(f"visible for roughly {duration:.0f}s")
    detail = "; ".join(bits) if bits else "only weak or sparse behavioral cues in the merged evidence"
    return (
        f"Scored {score:.1f}/100 as potential leadership. {detail.capitalize()}. "
        "(Summary derived from aggregated tracks — the LLM did not return a reasoning field.)"
    )


def _normalize_candidates(
    candidates: List[dict],
    agg: Dict[int, PersonEvidence],
) -> List[dict]:
    out: List[dict] = []
    for c in candidates:
        pid = c.get("person_id")
        if pid is None or int(pid) not in agg:
            continue
        pid = int(pid)
        score = float(c.get("leader_score", 0))
        r = c.get("reasoning")
        r = str(r).strip() if r is not None else ""
        if len(r) < 15:
            r = _reasoning_from_evidence(agg[pid], score)
        out.append({"person_id": pid, "leader_score": score, "reasoning": r})
    return out


def _fallback_heuristic(agg: Dict[int, PersonEvidence]) -> List[dict]:
    """Deterministic fallback when LLM is unavailable."""
    results = []
    for p in agg.values():
        obs = max(1, p.observations)
        duration = max(0.0, p.max_ts - (p.min_ts if p.min_ts != float("inf") else 0.0))
        score = (
            25 * (p.front_count / obs)
            + 25 * min(1.0, (0.4 * p.megaphone_count + 0.25 * p.microphone_count
                              + 0.2 * p.banner_count + 0.15 * p.flag_count) / obs)
            + 15 * min(1.0, p.gesture_total / (obs * 2))
            + 15 * min(1.0, obs / 20)
            + 10 * min(1.0, duration / 30)
            + 10 * min(1.0, (len(p.cameras) - 1) / 2)
        )
        results.append({
            "person_id": p.person_id,
            "leader_score": round(score, 1),
            "reasoning": _reasoning_from_evidence(p, round(score, 1)),
        })
    results.sort(key=lambda x: x["leader_score"], reverse=True)
    return results


def score_with_llm(
    description_rows: List[dict],
    model: str = "llama3.1:8b",
    ollama_url: str = "http://127.0.0.1:11434",
    timeout_sec: float = 120.0,
    top_k: int = 10,
) -> dict:
    """Score all detected people using the LLM as the judge.

    Returns a dict with:
      - candidates: ranked list of {person_id, leader_score, reasoning}
      - backend_used: "ollama" or "heuristic_fallback"
      - model: model name used
      - evidence_summary: per-person stats
    """
    agg = aggregate_evidence(description_rows)

    if not agg:
        return {
            "candidates": [],
            "backend_used": "none",
            "model": model,
            "evidence_summary": {},
        }

    context_blocks = []
    for pid in sorted(agg.keys()):
        context_blocks.append(person_evidence_to_context(agg[pid]))

    prompt = (
        "Below is the evidence collected from protest footage analysis.\n"
        "Score each person 0-100 on leadership likelihood.\n"
        "For EVERY person you MUST write a clear 'reasoning' field: explain WHY they received that score "
        "(which cues mattered: position, objects, gestures, duration, text descriptions).\n\n"
        + "\n\n".join(context_blocks)
        + "\n\nRespond with JSON only (array of objects with person_id, leader_score, reasoning)."
    )

    backend_used = "ollama"
    try:
        raw_response = _call_ollama(
            prompt=prompt, model=model,
            url=ollama_url, timeout_sec=timeout_sec,
        )
        candidates = _parse_llm_scores(raw_response)

        known_ids = set(agg.keys())
        candidates = [c for c in candidates if c.get("person_id") in known_ids]
        candidates = _normalize_candidates(candidates, agg)
        candidates.sort(key=lambda x: float(x.get("leader_score", 0)), reverse=True)

    except Exception as exc:
        print(f"[RAG-scorer] LLM call failed, using heuristic fallback: {exc}")
        candidates = _fallback_heuristic(agg)
        backend_used = "heuristic_fallback"

    candidates = candidates[:top_k]

    evidence_summary = {}
    for pid, p in agg.items():
        duration = max(0.0, p.max_ts - (p.min_ts if p.min_ts != float("inf") else 0.0))
        evidence_summary[pid] = {
            "observations": p.observations,
            "cameras": sorted(p.cameras),
            "front_count": p.front_count,
            "megaphone_count": p.megaphone_count,
            "banner_count": p.banner_count,
            "gesture_total": p.gesture_total,
            "duration_sec": round(duration, 1),
        }

    return {
        "candidates": candidates,
        "backend_used": backend_used,
        "model": model,
        "evidence_summary": evidence_summary,
    }


# ---------------------------------------------------------------------------
# File-based entrypoint (CLI & pipeline integration)
# ---------------------------------------------------------------------------

def load_description_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_rag_scoring(
    input_jsonl: str,
    output_jsonl: str,
    output_summary: str,
    top_k: int = 10,
    model: str = "llama3.1:8b",
    ollama_url: str = "http://127.0.0.1:11434",
    timeout_sec: float = 120.0,
) -> dict:
    rows = load_description_rows(input_jsonl)
    if not rows:
        raise RuntimeError(f"No description rows in {input_jsonl}")

    result = score_with_llm(
        description_rows=rows,
        model=model, ollama_url=ollama_url,
        timeout_sec=timeout_sec, top_k=top_k,
    )

    out_jsonl = Path(output_jsonl).resolve()
    out_summary = Path(output_summary).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in result["candidates"]:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "people_scored": len(result["candidates"]),
        "backend_used": result["backend_used"],
        "model": result["model"],
        "top_candidates": result["candidates"],
        "evidence_summary": result["evidence_summary"],
    }
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"RAG scoring completed (backend: {result['backend_used']})")
    print(f"Scores: {out_jsonl}")
    print(f"Summary: {out_summary}")

    return result
