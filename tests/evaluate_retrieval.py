"""Evaluate CinemaSearch scene retrieval (BM25 vs TF-IDF)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from irsearch.evaluation import evaluate_rankings
from irsearch.pipeline import load
from irsearch.search import SearchEngine

DEFAULT_QRELS = Path(__file__).with_name("relevance_judgments_full.json")
METHODS = ("bm25", "tfidf")


def scene_id(movie_id, scene_idx) -> str:
    return f"{movie_id}:{scene_idx}"



# PHASE 1: LOAD JUDGMENTS (chunk IDs are converted to scene IDs)


def load_judgments(path: Path) -> dict[str, list]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or not all(
        isinstance(ids, list) for ids in data.values()
    ):
        raise ValueError("Judgments must be {query: [ids, ...]}.")

    return data


def to_scene_judgments(raw: dict[str, list], store) -> dict[str, set[str]]:
    legacy_chunk_ids = sorted({
        item
        for ids in raw.values()
        for item in ids
        if isinstance(item, int) and not isinstance(item, bool)
    })

    chunk_to_scene = {}
    if legacy_chunk_ids:
        for chunk in store.get_chunks(legacy_chunk_ids):
            chunk_to_scene[chunk["chunk_id"]] = scene_id(
                chunk["movie_id"], chunk["scene_idx"]
            )

    judgments = {}
    for query, ids in raw.items():
        scenes = set()
        for item in ids:
            if isinstance(item, bool):
                raise ValueError(f"Invalid ID for query: {query}")
            if isinstance(item, int):
                if item in chunk_to_scene:
                    scenes.add(chunk_to_scene[item])
                else:
                    print(f"Warning: chunk {item} not found, skipped.")
            elif isinstance(item, str):
                scenes.add(item)
            else:
                raise ValueError(f"Invalid ID for query: {query}")
        judgments[query] = scenes

    return judgments


def rewrite_qrels(path: Path, judgments: dict[str, set[str]]) -> None:
    data = {query: sorted(ids) for query, ids in judgments.items()}
    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Judgments rewritten with scene IDs: {path}")



# PHASE 2: RUN SEARCHES


def run_searches(engines, queries, top_k):
    rankings = {method: {} for method in engines}
    results = {method: {} for method in engines}

    for method, engine in engines.items():
        for query in queries:
            found = engine.search(query, top_k=top_k)
            results[method][query] = found
            rankings[method][query] = [
                scene_id(r["movie_id"], r["scene_idx"]) for r in found
            ]

    return rankings, results



# PHASE 3: SHOW CANDIDATES FOR MANUAL JUDGING


def show_candidates(queries, results, snippet=350) -> None:
    for query in queries:
        pooled = {}
        for method, per_query in results.items():
            for r in per_query[query]:
                sid = scene_id(r["movie_id"], r["scene_idx"])
                entry = pooled.setdefault(sid, {"result": r, "methods": []})
                entry["methods"].append(method)

        print("\n" + "=" * 75)
        print(f"QUERY: {query}  ({len(pooled)} candidate scenes)")
        print("=" * 75)

        for sid, entry in pooled.items():
            r = entry["result"]
            text = r["text"].replace("\n", " ")[:snippet]
            print(f"\nScene ID: {sid}  [{', '.join(entry['methods'])}]")
            print(f"{r['title']} ({r['year']}) - {r.get('heading')}")
            print(f"{text}...")



# PHASE 4: METRICS AND REPORT


def print_report(reports, rankings, judgments, k) -> None:
    methods = list(reports)

    print("\nPER QUERY  (AP@k | unjudged results in top-k)")
    print("-" * 75)

    for query, relevant in judgments.items():
        cells = []
        for method in methods:
            ap = next(
                row[f"ap@{k}"]
                for row in reports[method]["per_query"]
                if row["query"] == query
            )
            unjudged = sum(
                sid not in relevant for sid in rankings[method][query]
            )
            cells.append(f"{method}: {ap:.4f} | {unjudged}")
        print(f"{query[:45]:<46}" + "   ".join(cells))
        print(f"{'':<46}relevant scenes: {len(relevant)}")

    print("\nOVERALL")
    print("-" * 75)
    print(f"{'method':<8}{'P@k':>9}{'R@k':>9}{'F1@k':>9}{'MAP@k':>9}")

    for method in methods:
        rep = reports[method]
        print(
            f"{method:<8}"
            f"{rep[f'precision@{k}']:>9.4f}"
            f"{rep[f'recall@{k}']:>9.4f}"
            f"{rep[f'f1@{k}']:>9.4f}"
            f"{rep[f'map@{k}']:>9.4f}"
        )

    print(f"\nk = {k}, evaluated queries = {reports[methods[0]]['evaluated_queries']}")



# MAIN


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CinemaSearch scene retrieval."
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--corpus", default="out_full")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Print pooled candidate scenes for queries without judgments.",
    )
    parser.add_argument(
        "--rewrite-qrels",
        action="store_true",
        help="Save the judgments file using scene IDs.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")

    raw = load_judgments(args.qrels)
    index, store, preprocessor, _ = load(args.corpus)

    try:
        judgments = to_scene_judgments(raw, store)

        if args.rewrite_qrels:
            rewrite_qrels(args.qrels, judgments)

        engines = {
            method: SearchEngine(index, store, preprocessor, ranking_method=method)
            for method in args.methods
        }

        rankings, results = run_searches(engines, list(judgments), args.top_k)

        if args.show_candidates:
            empty = [q for q, ids in judgments.items() if not ids]
            show_candidates(empty, results)

        active = {q: ids for q, ids in judgments.items() if ids}

        if not active:
            print("No judged queries yet. Add scene IDs to the judgments file.")
            return

        reports = {
            method: evaluate_rankings(rankings[method], active, k=args.top_k)
            for method in engines
        }

        print_report(reports, rankings, active, args.top_k)

    finally:
        store.close()


if __name__ == "__main__":
    main()