"""
analyze_failures.py

Reads results/comparison.json (written by query_test.py) and finds
regressions: facts the baseline (full-precision) index retrieved correctly
but a compressed index missed. Prints them grouped by index, so you can see
exactly which numbers/facts degrade under quantization instead of just the
aggregate recall%.

Usage: python analyze_failures.py [--fact-type number] [--index turbovec_b2]
"""

import argparse
import json
import os

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "comparison.json")
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ground_truth.json")


def load_fact_types():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)["facts"]
    return {f["id"]: f["fact_type"] for f in facts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fact-type", default=None, help="filter to one fact_type (id/name/date/number)")
    parser.add_argument("--index", default="turbovec_b2", help="compressed index to diff against baseline")
    args = parser.parse_args()

    fact_types = load_fact_types()

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "baseline" not in data or args.index not in data:
        print(f"Need both 'baseline' and '{args.index}' in {RESULTS_PATH}")
        return

    baseline_by_fact = {r["fact_id"]: r for r in data["baseline"]["raw_results"]}
    compressed_by_fact = {r["fact_id"]: r for r in data[args.index]["raw_results"]}

    regressions = []
    also_broken = []  # failed on both baseline and compressed — not a quantization regression

    for fact_id, base_r in baseline_by_fact.items():
        if args.fact_type and fact_types.get(fact_id) != args.fact_type:
            continue

        comp_r = compressed_by_fact.get(fact_id)
        if comp_r is None:
            continue

        if base_r["hit"] and not comp_r["hit"]:
            regressions.append((fact_id, base_r, comp_r))
        elif not base_r["hit"] and not comp_r["hit"]:
            also_broken.append((fact_id, base_r, comp_r))

    print(f"=== Regressions: baseline hit, {args.index} missed ({len(regressions)}) ===\n")
    for fact_id, base_r, comp_r in regressions:
        print(f"[{fact_id}] {base_r['question']}")
        print(f"    expected chunk: {base_r['expected_chunk_id']}")
        print(f"    baseline retrieved:      {base_r['retrieved_chunk_ids']}")
        print(f"    {args.index} retrieved: {comp_r['retrieved_chunk_ids']}")
        print()

    print(f"\n=== Also broken on baseline too (not a quantization issue) ({len(also_broken)}) ===\n")
    for fact_id, base_r, comp_r in also_broken:
        print(f"[{fact_id}] {base_r['question']}  (expected {base_r['expected_chunk_id']})")


if __name__ == "__main__":
    main()
