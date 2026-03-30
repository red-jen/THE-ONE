"""RAG chatbot — the LLM reasons over evidence and answers questions.

Flow:
1. Load description evidence from the memory store
2. Load pre-aggregated signals (if available)
3. Retrieve relevant evidence for the user's question
4. Ask the LLM to answer using candidates + evidence as context
5. Return structured answer with scores, reasoning, and sources
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List
from urllib import error, request


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _overlap_score(q_tokens: set[str], text: str) -> float:
    doc_tokens = _tokenize(text)
    if not q_tokens or not doc_tokens:
        return 0.0
    overlap = len(q_tokens & doc_tokens)
    return overlap / max(1.0, len(q_tokens) ** 0.5 * len(doc_tokens) ** 0.5)


def _load_scores(path: str) -> List[dict]:
    rows: List[dict] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(
        key=lambda r: float(r.get("leader_score", r.get("suspicion_score", 0))),
        reverse=True,
    )
    return rows


def _load_simple_store(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("documents", [])


def _retrieve_evidence(query: str, docs: List[dict], top_k: int) -> List[dict]:
    q_tokens = _tokenize(query)
    scored = []
    for doc in docs:
        text = str(doc.get("text", ""))
        score = _overlap_score(q_tokens, text)
        if score > 0:
            scored.append({
                "relevance": round(score, 6),
                "doc_id": doc.get("doc_id"),
                "text": text,
                "metadata": doc.get("metadata", {}),
            })
    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored[:max(1, top_k)]


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _build_context(
    candidates: List[dict],
    evidence: List[dict],
) -> str:
    lines: List[str] = []

    lines.append("=== RANKED CANDIDATES (scored by AI analysis) ===")
    if not candidates:
        lines.append("No candidates scored yet.")
    for i, c in enumerate(candidates, 1):
        score = c.get("leader_score", c.get("suspicion_score", "?"))
        reasoning = c.get("reasoning", "")
        lines.append(f"  #{i}  person_id={c.get('person_id')}  leader_score={score}")
        if reasoning:
            lines.append(f"       reasoning: {reasoning}")

    lines.append("")
    lines.append("=== RETRIEVED EVIDENCE ===")
    if not evidence:
        lines.append("No matching evidence found.")
    for i, ev in enumerate(evidence, 1):
        meta = ev.get("metadata", {})
        lines.append(
            f"  [{i}] person_id={meta.get('person_id')} "
            f"cam={meta.get('camera_id')} frame={meta.get('frame_idx')} "
            f"relevance={ev.get('relevance')}"
        )
        lines.append(f"       {ev.get('text', '')[:300]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = """You are an expert protest intelligence analyst.
You answer questions about detected people in protest footage.
Use ONLY the provided context (candidates + evidence). Do not invent facts.
Structure your answer as:
1. Direct answer to the question
2. Supporting evidence from the observations
3. Confidence assessment and caveats"""


def _generate_ollama(
    question: str,
    context: str,
    model: str,
    url: str,
    timeout_sec: float,
) -> str:
    prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer based only on the context above."
    )
    # Preferred path: LangChain + Ollama.
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOllama(
            model=model,
            base_url=url,
            temperature=0.3,
            num_predict=1024,
            timeout=timeout_sec,
        )
        resp = llm.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=prompt)])
        text = str(getattr(resp, "content", "")).strip()
        if text:
            return text
    except Exception:
        # Fallback to direct Ollama HTTP call if LangChain is unavailable.
        pass

    payload = {
        "model": model,
        "system": ANSWER_SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 1024},
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
    text = str(parsed.get("response", "")).strip()
    if not text:
        raise RuntimeError("Empty response from Ollama")
    return text


def _generate_template(
    question: str,
    candidates: List[dict],
    evidence: List[dict],
) -> str:
    lines = [f"Question: {question}", ""]
    if candidates:
        lines.append("Based on the analysis, the most likely leaders are:")
        for i, c in enumerate(candidates[:5], 1):
            score = c.get("leader_score", c.get("suspicion_score", "?"))
            reasoning = c.get("reasoning", "No reasoning available")
            lines.append(f"  #{i} Person {c.get('person_id')} — score {score}/100")
            lines.append(f"      {reasoning}")
    else:
        lines.append("No candidates have been scored yet.")

    if evidence:
        lines.append("")
        lines.append("Supporting evidence:")
        for i, ev in enumerate(evidence[:5], 1):
            meta = ev.get("metadata", {})
            lines.append(f"  [{i}] Person {meta.get('person_id')} (cam {meta.get('camera_id')}): "
                         f"{ev.get('text', '')[:200]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_query(
    question: str,
    scores_jsonl: str,
    simple_store_json: str,
    top_leaders: int = 5,
    top_evidence: int = 8,
    generator_backend: str = "template",
    ollama_model: str = "llama3.1:8b",
    ollama_url: str = "http://127.0.0.1:11434",
    ollama_timeout_sec: float = 60.0,
) -> dict:
    """Answer a natural-language question using scored candidates + evidence."""

    candidates = _load_scores(scores_jsonl)[:top_leaders]
    docs = _load_simple_store(simple_store_json)
    evidence = _retrieve_evidence(question, docs, top_k=top_evidence)

    backend_used = generator_backend
    context = _build_context(candidates, evidence)

    if generator_backend == "ollama":
        try:
            answer_text = _generate_ollama(
                question=question,
                context=context,
                model=ollama_model,
                url=ollama_url,
                timeout_sec=ollama_timeout_sec,
            )
        except (RuntimeError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            backend_used = "template_fallback"
            answer_text = _generate_template(question, candidates, evidence)
            answer_text += f"\n\n[Note: LLM unavailable ({exc}), used template fallback]"
    else:
        answer_text = _generate_template(question, candidates, evidence)

    return {
        "question": question,
        "answer": answer_text,
        "top_candidates": candidates,
        "evidence_used": evidence,
        "generator_backend": backend_used,
        "model": ollama_model if "ollama" in backend_used else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG chatbot — ask questions about protest analysis")
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--scores-jsonl", type=str, default="outputs/scores/leader_scores.jsonl")
    p.add_argument("--simple-store", type=str, default="outputs/memory/simple_store.json")
    p.add_argument("--top-leaders", type=int, default=5)
    p.add_argument("--top-evidence", type=int, default=8)
    p.add_argument("--generator-backend", choices=["template", "ollama"], default="template")
    p.add_argument("--ollama-model", type=str, default="llama3.1:8b")
    p.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434")
    p.add_argument("--ollama-timeout-sec", type=float, default=60.0)
    p.add_argument("--output-json", type=str, default="outputs/rag/answer.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = answer_query(
        question=args.question,
        scores_jsonl=args.scores_jsonl,
        simple_store_json=args.simple_store,
        top_leaders=args.top_leaders,
        top_evidence=args.top_evidence,
        generator_backend=args.generator_backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        ollama_timeout_sec=args.ollama_timeout_sec,
    )
    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Answer saved to {out}")
    print(result["answer"])


if __name__ == "__main__":
    main()
