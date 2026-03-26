from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _simple_overlap_score(query_tokens: set[str], text: str) -> float:
    doc_tokens = _tokenize(text)
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(doc_tokens))
    return overlap / max(1.0, len(query_tokens) ** 0.5 * len(doc_tokens) ** 0.5)


def _load_scores(scores_jsonl: str) -> List[dict]:
    rows: List[dict] = []
    with Path(scores_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: float(r.get("leader_score", 0.0)), reverse=True)
    return rows


def _load_simple_store(simple_store_json: str) -> List[dict]:
    with Path(simple_store_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("documents", [])


def _retrieve_evidence(query: str, docs: List[dict], top_k: int) -> List[dict]:
    q_tokens = _tokenize(query)
    scored: List[dict] = []
    for doc in docs:
        text = str(doc.get("text", ""))
        score = _simple_overlap_score(q_tokens, text)
        if score <= 0:
            continue
        scored.append(
            {
                "score": round(score, 6),
                "doc_id": doc.get("doc_id"),
                "text": text,
                "metadata": doc.get("metadata", {}),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[: max(1, top_k)]


def build_answer(question: str, scored_people: List[dict], evidence: List[dict], top_leaders: int) -> dict:
    top_candidates = scored_people[: max(1, top_leaders)]

    answer_lines: List[str] = []
    answer_lines.append(f"Question: {question}")

    if not top_candidates:
        answer_lines.append("No scored candidates are available yet.")
    else:
        answer_lines.append("Top leader candidates:")
        for idx, person in enumerate(top_candidates, start=1):
            person_id = person.get("person_id")
            leader_score = person.get("leader_score")
            components = person.get("components", {})
            answer_lines.append(
                f"{idx}. Person {person_id} with score {leader_score} "
                f"(front={components.get('front_presence', 0)}, "
                f"objects={components.get('object_signal', 0)}, "
                f"gesture={components.get('gesture_signal', 0)})."
            )

    if evidence:
        answer_lines.append("Relevant evidence snippets:")
        for idx, ev in enumerate(evidence, start=1):
            meta = ev.get("metadata", {})
            answer_lines.append(
                f"- [{idx}] score={ev.get('score')} person={meta.get('person_id')} "
                f"camera={meta.get('camera_id')} frame={meta.get('frame_idx')}"
            )
            answer_lines.append(f"  {ev.get('text')}")
    else:
        answer_lines.append("No lexical evidence matched the question.")

    return {
        "question": question,
        "top_candidates": top_candidates,
        "evidence": evidence,
        "answer": "\n".join(answer_lines),
    }


def answer_query(
    question: str,
    scores_jsonl: str,
    simple_store_json: str,
    top_leaders: int = 3,
    top_evidence: int = 5,
) -> dict:
    scored_people = _load_scores(scores_jsonl)
    docs = _load_simple_store(simple_store_json)
    evidence = _retrieve_evidence(question, docs, top_k=top_evidence)
    return build_answer(
        question=question,
        scored_people=scored_people,
        evidence=evidence,
        top_leaders=top_leaders,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RAG-style query interface over scores + memory store")
    parser.add_argument("--question", type=str, required=True, help="Natural language question")
    parser.add_argument("--scores-jsonl", type=str, default="outputs/scores/test_run_scores.jsonl")
    parser.add_argument("--simple-store", type=str, default="outputs/memory/test_run/simple_store.json")
    parser.add_argument("--top-leaders", type=int, default=3)
    parser.add_argument("--top-evidence", type=int, default=5)
    parser.add_argument("--output-json", type=str, default="outputs/rag/query_answer.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = answer_query(
        question=args.question,
        scores_jsonl=args.scores_jsonl,
        simple_store_json=args.simple_store,
        top_leaders=args.top_leaders,
        top_evidence=args.top_evidence,
    )

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("RAG-style query completed")
    print(f"Output JSON: {out_path}")
    print("Answer:")
    print(result["answer"])


if __name__ == "__main__":
    main()
