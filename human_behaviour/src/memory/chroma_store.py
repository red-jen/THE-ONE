from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from pathlib import Path
from typing import List, Tuple


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def row_to_document(row: dict) -> str:
    desc = str(row.get("description", ""))
    signals = row.get("signals", {})
    signal_bits = [
        f"megaphone={signals.get('has_megaphone', False)}",
        f"banner={signals.get('has_banner', False)}",
        f"flag={signals.get('has_flag', False)}",
        f"microphone={signals.get('has_microphone', False)}",
        f"gesture={signals.get('gesture_score', 0)}",
        f"position={signals.get('position_hint', 'unknown')}",
    ]
    meta = f"person_id={row.get('person_id')} camera_id={row.get('camera_id')} frame_idx={row.get('frame_idx')}"
    return f"{desc} {' '.join(signal_bits)} {meta}".strip()


def load_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_simple_store(rows: List[dict]) -> dict:
    docs = []
    for idx, row in enumerate(rows):
        docs.append(
            {
                "doc_id": f"desc_{idx}",
                "text": row_to_document(row),
                "tokens": sorted(tokenize(row_to_document(row))),
                "metadata": {
                    "person_id": row.get("person_id"),
                    "camera_id": row.get("camera_id"),
                    "frame_idx": row.get("frame_idx"),
                    "timestamp_sec": row.get("timestamp_sec"),
                    "bbox": row.get("bbox"),
                },
            }
        )
    return {"backend": "simple", "documents": docs}


def score_overlap(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(doc_tokens))
    norm = math.sqrt(len(query_tokens)) * math.sqrt(max(1, len(doc_tokens)))
    return float(overlap / max(norm, 1e-9))


def query_simple_store(store: dict, query: str, top_k: int) -> List[dict]:
    query_tokens = tokenize(query)
    scored = []
    for doc in store.get("documents", []):
        score = score_overlap(query_tokens, set(doc.get("tokens", [])))
        if score <= 0:
            continue
        scored.append(
            {
                "doc_id": doc["doc_id"],
                "score": round(score, 6),
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[: max(1, top_k)]


def run_index(input_jsonl: str, store_dir: str, backend: str) -> None:
    rows = load_rows(input_jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in {input_jsonl}")

    out_dir = Path(store_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_backend = backend
    if backend == "auto":
        try:
            importlib.import_module("chromadb")

            selected_backend = "chroma"
        except Exception:
            selected_backend = "simple"

    if selected_backend == "chroma":
        try:
            chromadb = importlib.import_module("chromadb")
        except Exception as exc:
            raise RuntimeError("Chroma backend requested but chromadb is not installed.") from exc

        client = chromadb.PersistentClient(path=str(out_dir / "chroma_db"))
        collection = client.get_or_create_collection(name="descriptions")

        ids = []
        docs = []
        metas = []
        for idx, row in enumerate(rows):
            ids.append(f"desc_{idx}")
            docs.append(row_to_document(row))
            metas.append(
                {
                    "person_id": int(row.get("person_id", -1)),
                    "camera_id": str(row.get("camera_id", "unknown")),
                    "frame_idx": int(row.get("frame_idx", -1)),
                    "timestamp_sec": float(row.get("timestamp_sec", 0.0)),
                }
            )

        if ids:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)

        summary = {
            "backend": "chroma",
            "documents_indexed": len(ids),
            "store_dir": str(out_dir),
            "collection": "descriptions",
        }
        with (out_dir / "index_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("=" * 60)
        print("Memory indexing completed")
        print("Backend: chroma")
        print(f"Store dir: {out_dir}")
        print(f"Documents indexed: {len(ids)}")
        return

    if selected_backend == "simple":
        store = build_simple_store(rows)
        with (out_dir / "simple_store.json").open("w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False)

        summary = {
            "backend": "simple",
            "documents_indexed": len(store.get("documents", [])),
            "store_dir": str(out_dir),
        }
        with (out_dir / "index_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("=" * 60)
        print("Memory indexing completed")
        print("Backend: simple")
        print(f"Store dir: {out_dir}")
        print(f"Documents indexed: {len(store.get('documents', []))}")
        return

    raise ValueError(f"Unsupported backend: {selected_backend}")


def run_query(store_dir: str, query: str, top_k: int) -> None:
    base = Path(store_dir).resolve()

    simple_path = base / "simple_store.json"
    chroma_path = base / "chroma_db"

    if simple_path.exists():
        with simple_path.open("r", encoding="utf-8") as f:
            store = json.load(f)
        results = query_simple_store(store=store, query=query, top_k=top_k)

        print("=" * 60)
        print("Query results")
        print("Backend: simple")
        print(f"Query: {query}")
        print(f"Top-k: {top_k}")
        for idx, item in enumerate(results, start=1):
            print(f"[{idx}] score={item['score']} id={item['doc_id']} meta={item['metadata']}")
            print(f"     text={item['text']}")
        return

    if chroma_path.exists():
        try:
            chromadb = importlib.import_module("chromadb")
        except Exception as exc:
            raise RuntimeError("Chroma store detected but chromadb is not installed.") from exc

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(name="descriptions")

        result = collection.query(query_texts=[query], n_results=max(1, top_k))
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0] if "distances" in result else [None] * len(ids)

        print("=" * 60)
        print("Query results")
        print("Backend: chroma")
        print(f"Query: {query}")
        print(f"Top-k: {top_k}")
        for idx, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            print(f"[{idx}] id={doc_id} distance={dist} meta={meta}")
            print(f"     text={doc}")
        return

    raise RuntimeError(f"No known store found in {base}. Run index first.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Description memory index and query (Chroma or fallback)")
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="Index description JSONL into memory store")
    idx.add_argument("--input-jsonl", type=str, required=True)
    idx.add_argument("--store-dir", type=str, default="outputs/memory_store")
    idx.add_argument("--backend", type=str, choices=["auto", "chroma", "simple"], default="auto")

    qry = sub.add_parser("query", help="Query indexed store")
    qry.add_argument("--store-dir", type=str, default="outputs/memory_store")
    qry.add_argument("--query", type=str, required=True)
    qry.add_argument("--top-k", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "index":
        run_index(input_jsonl=args.input_jsonl, store_dir=args.store_dir, backend=args.backend)
        return

    if args.command == "query":
        run_query(store_dir=args.store_dir, query=args.query, top_k=args.top_k)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
