"""
analyze_by_section.py

Breaks recall@k down by REPORT SECTION (financial / incident / personnel /
server) instead of just fact_type, so we can check specific claims like
"financial numbers are worse than other numbers" directly, instead of
guessing from the fact_type-only aggregate.

Chunk ranges (from fake_report.py's build order):
  C001-C006  incidents
  C007-C010  financial summaries
  C011-C016  personnel records
  C017-C022  server specs
"""

import json
import os

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "comparison.json")
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ground_truth.json")

SECTIONS = {
    "incident":   {f"C{i:03d}" for i in range(1, 7)},
    "financial":  {f"C{i:03d}" for i in range(7, 11)},
    "personnel":  {f"C{i:03d}" for i in range(11, 17)},
    "server":     {f"C{i:03d}" for i in range(17, 23)},
}


def section_of(chunk_id):
    for name, ids in SECTIONS.items():
        if chunk_id in ids:
            return name
    return "unknown"


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for index_name in ("baseline", "turbovec_b4", "turbovec_b2"):
        if index_name not in data:
            continue

        by_section = {}
        for r in data[index_name]["raw_results"]:
            sec = section_of(r["expected_chunk_id"])
            by_section.setdefault(sec, {"hits": 0, "total": 0})
            by_section[sec]["total"] += 1
            if r["hit"]:
                by_section[sec]["hits"] += 1

        print(f"\n=== {index_name} — recall@3 by section ===")
        for sec in ("incident", "financial", "personnel", "server"):
            c = by_section.get(sec, {"hits": 0, "total": 0})
            if c["total"] == 0:
                continue
            recall = c["hits"] / c["total"]
            print(f"{sec:<10} {c['hits']}/{c['total']:<6} {recall:.1%}")


if __name__ == "__main__":
    main()
