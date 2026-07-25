"""
query_test.py

Reads data/ground_truth.json (fact questions + correct chunk_id) and the
indexes built by embed_and_index.py:

  - indexes/baseline.npz     - raw float32 vectors + chunk_ids (brute-force
                                 cosine search, done here in numpy)
  - indexes/turbovec_b4.tvim - turbovec.IdMapIndex, bit_width=4
  - indexes/turbovec_b2.tvim - turbovec.IdMapIndex, bit_width=2

For each fact question:
  1. Embed the question with the same local sentence-transformers model
     used to build the indexes.
  2. Retrieve top-k nearest chunks from each index.
  3. Check whether the correct chunk_id (from ground truth) is in top-k.

Aggregates results by fact_type (id / name / date / number) so we can see
whether quantization degrades some fact types faster than others — that's
the actual question this project is testing.

Output: results/comparison.json + a printed summary table.
"""

import argparse
import json
import os

import numpy as np
import turbovec
from sentence_transformers import SentenceTransformer

from embed_and_index import MODEL_PATH, DATA_DIR, INDEX_DIR
from chunk_id_codec import uint64_to_chunk_id

GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "comparison.json")

TOP_K = 3  # how many nearest chunks count as a "hit"

INDEXES = {
    "baseline": os.path.join(INDEX_DIR, "baseline.npz"),
    "turbovec_b4": os.path.join(INDEX_DIR, "turbovec_b4.tvim"),
    "turbovec_b2": os.path.join(INDEX_DIR, "turbovec_b2.tvim"),
}


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["facts"]


# ---------------------------------------------------------------------------
# Search backends — each returns, for a batch of query vectors, a list of
# top-k chunk_id lists (one list per query, best match first).
# ---------------------------------------------------------------------------

def make_baseline_searcher(path):
    npz = np.load(path, allow_pickle=True)
    chunk_ids = npz["chunk_ids"]
    vectors = npz["vectors"].astype(np.float32)

    def search(query_vecs, k):
        sims = query_vecs @ vectors.T  # (n_queries, n_chunks), cosine via dot
        order = np.argsort(-sims, axis=1)[:, :k]
        return [[str(chunk_ids[i]) for i in row] for row in order]

    return search


def make_turbovec_searcher(path):
    index = turbovec.IdMapIndex.load(path)

    def search(query_vecs, k):
        scores, ids = index.search(query_vecs, k)
        return [[uint64_to_chunk_id(i) for i in row] for row in ids]

    return search


def build_searcher(name, path):
    if name == "baseline":
        return make_baseline_searcher(path)
    return make_turbovec_searcher(path)


# ---------------------------------------------------------------------------

def evaluate_index(name, search_fn, facts, query_vecs):
    """Runs every ground-truth question against one index's search_fn."""
    by_type = {}
    raw_results = []

    retrieved_batch = search_fn(query_vecs, TOP_K)

    for fact, retrieved in zip(facts, retrieved_batch):
        hit = fact["chunk_id"] in retrieved

        ftype = fact["fact_type"]
        by_type.setdefault(ftype, {"hits": 0, "total": 0})
        by_type[ftype]["total"] += 1
        if hit:
            by_type[ftype]["hits"] += 1

        raw_results.append({
            "fact_id": fact["id"],
            "question": fact["question"],
            "expected_chunk_id": fact["chunk_id"],
            "retrieved_chunk_ids": retrieved,
            "hit": hit,
        })

    overall_hits = sum(v["hits"] for v in by_type.values())
    overall_total = sum(v["total"] for v in by_type.values())

    print(f"\n=== {name} ===")
    print(f"{'fact_type':<10} {'hits/total':<12} {'recall@' + str(TOP_K):<10}")
    for ftype, counts in sorted(by_type.items()):
        recall = counts["hits"] / counts["total"] if counts["total"] else 0.0
        label = f"{counts['hits']}/{counts['total']}"
        print(f"{ftype:<10} {label:<12} {recall:.2%}")
    overall_recall = overall_hits / overall_total if overall_total else 0.0
    label = f"{overall_hits}/{overall_total}"
    print(f"{'OVERALL':<10} {label:<12} {overall_recall:.2%}")

    return {
        "by_fact_type": by_type,
        "overall_hits": overall_hits,
        "overall_total": overall_total,
        "overall_recall": overall_recall,
        "raw_results": raw_results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to a local sentence-transformers snapshot folder. Must "
             "match the model used in embed_and_index.py. Falls back to the "
             "QUANTCHECK_MODEL_PATH env var if not given.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path

    if not model_path:
        raise SystemExit(
            "No model path given. Pass --model-path /path/to/snapshot, or "
            "set the QUANTCHECK_MODEL_PATH environment variable."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading ground truth from {GROUND_TRUTH_PATH}")
    facts = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"{len(facts)} fact questions loaded")

    print(f"Loading model from local snapshot: {model_path}")
    model = SentenceTransformer(model_path, local_files_only=True)

    print("Embedding questions...")
    questions = [f["question"] for f in facts]
    query_vecs = model.encode(
        questions,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    all_results = {}

    for name, path in INDEXES.items():
        if not os.path.exists(path):
            print(f"Skipping {name} — {path} not found (run embed_and_index.py first)")
            continue

        search_fn = build_searcher(name, path)
        all_results[name] = evaluate_index(name, search_fn, facts, query_vecs)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {RESULTS_PATH}")

    if "baseline" in all_results:
        others = [n for n in all_results if n != "baseline"]
        print(f"\n=== Summary: recall@{TOP_K} vs baseline, by fact_type ===")
        header = f"{'fact_type':<10} {'baseline':<10}" + "".join(f"{n:<14}" for n in others)
        print(header)
        baseline_by_type = all_results["baseline"]["by_fact_type"]
        for ftype in sorted(baseline_by_type):
            base_c = baseline_by_type[ftype]
            base_r = base_c["hits"] / base_c["total"] if base_c["total"] else 0.0
            row = f"{ftype:<10} {base_r:.1%}".ljust(21)
            for name in others:
                c = all_results[name]["by_fact_type"].get(ftype, {"hits": 0, "total": 1})
                r = c["hits"] / c["total"] if c["total"] else 0.0
                row += f"{r:.1%}".ljust(14)
            print(row)


if __name__ == "__main__":
    main()
