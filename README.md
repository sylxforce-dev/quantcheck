# quantcheck
 
**Does TurboVec (TurboQuant-based vector compression) hold up on exact-fact
retrieval, or does it only work for fuzzy semantic similarity?**
 
A synthetic-report RAG test: 22 chunks, 104 fact-level questions (revenue
figures, ticket IDs, dates, names), retrieval compared across full
precision vs. TurboVec bit_width=4 vs. bit_width=2 (TurboVec's max
compression).
 
## TL;DR
 
| fact_type | baseline | turbovec_b4 | turbovec_b2 |
|-----------|----------|-------------|-------------|
| number    | 87.0%    | 84.8%       | 80.4%       |
| **overall** | **79.8%** | **80.8%** | **76.9%** |
 
- Numeric facts degrade with heavier compression — but the actual cause
  isn't "numbers are hard," it's **templated, near-duplicate chunks**
  (e.g. repeated server-spec records) where a number is the *only*
  differentiator. Quantization noise erases that thin margin.
- Baseline itself only hits 79.8% recall@3 — semantic embedding search is
  a mediocre exact-fact retriever even *before* any compression.
Full writeup, methodology, and the "when is compression actually safe"
framework: **[quantcheck/ANALYSIS.md](quantcheck/ANALYSIS.md)**
 
## What's in this repo
 
| File | Purpose |
|---|---|
| `quantcheck/fake_report.py` | Generates the synthetic report + ground-truth facts |
| `quantcheck/embed_and_index.py` | Builds baseline + TurboVec (b4/b2) indexes |
| `quantcheck/query_test.py` | Runs all 104 questions against all 3 indexes, prints recall@3 |
| `quantcheck/analyze_failures.py` | Shows exactly which questions regress under compression |
| `quantcheck/analyze_by_section.py` | Breaks recall down by report section instead of fact_type |
| `quantcheck/chunk_id_codec.py` | Helper module (not a script) — chunk_id ↔ uint64 conversion for TurboVec |
 
## Quick start
 
```bash
cd quantcheck
pip install -r requirements.txt
 
python fake_report.py
python embed_and_index.py --model-path /path/to/local/sentence-transformers/snapshot
python query_test.py
```
 
Needs a local sentence-transformers snapshot (offline, no HF calls at test
time) — see [quantcheck/ANALYSIS.md](quantcheck/ANALYSIS.md#setup) for how
to get one and full setup/reproduction details, including
`analyze_failures.py` / `analyze_by_section.py` usage and swapping in your
own data.
 
