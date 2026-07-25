# Full analysis: quantcheck

## Why this test exists

TurboVec sells itself on preserving semantic similarity under aggressive
quantization — good enough to still find "the doc about X" after 8x
compression. But RAG systems don't just retrieve *topics*, they retrieve
*facts*: exact revenue figures, ticket IDs, dates, employee names. Those are
a much narrower target in embedding space than "documents about the same
subject." I wanted to know whether the same compression that preserves
semantic recall also preserves the ability to find the *one* chunk containing
a specific number, or whether precision degrades faster than similarity does.

## Setup

**1. Install dependencies**

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**2. Get a local sentence-transformers model snapshot**

Every script here runs fully offline (`local_files_only=True`) — nothing
ever gets downloaded from Hugging Face at test time. You need a local copy
of a sentence-transformers model *before* running anything. Two ways to get
one:

- **You already have one cached.** If you've ever run
  `SentenceTransformer("all-MiniLM-L6-v2")` on this machine before (in any
  project), it's already sitting at:
  `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>/`
  (Windows: `C:\Users\<you>\.cache\huggingface\hub\...`). Use that folder
  path directly.
- **You don't have one yet.** On a machine *with* internet access, run:
  ```bash
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
  ```
  This downloads it once into the same cache path above. From then on,
  point `--model-path` at that folder and you never need internet again —
  that's the whole "local/offline" point of this project.

You're pointing at the **snapshot folder itself** (the one containing
`config.json`, `model.safetensors`, etc.), not the model name string.

**3. Know the TurboVec bit_width limit**

`turbovec.IdMapIndex` only accepts `bit_width` of **2, 3, or 4** —
`bit_width=1` and `bit_width=8` both raise `ValueError`. This project tests
4 (light compression) and 2 (TurboVec's actual maximum compression, not an
arbitrary "aggressive" choice) so the results already cover the full range
TurboVec supports.

## Method

1. `fake_report.py` generates a synthetic 22-chunk report (incidents,
   financial summaries, personnel records, server specs) with known ground
   truth — 104 fact-level questions (`data/ground_truth.json`), each tagged
   `id` / `name` / `date` / `number` and pointing at the exact source chunk.
2. `embed_and_index.py` embeds all chunks locally (`all-MiniLM-L6-v2`, no
   network calls) and builds three indexes:
   - `baseline.npz` — full-precision vectors, brute-force cosine search
   - `turbovec_b4.tvim` — TurboVec `IdMapIndex`, bit_width=4
   - `turbovec_b2.tvim` — TurboVec `IdMapIndex`, bit_width=2

   `turbovec.IdMapIndex` requires external ids as `uint64`, but our chunks
   are labeled `"C001"`, `"C022"`, etc. `chunk_id_codec.py` is the one place
   that converts between the two (`"C001" <-> uint64(1)`), so both this
   script (writing ids in) and `query_test.py` (reading ids back out) stay
   in sync without duplicating that logic.
3. `query_test.py` embeds all 104 questions, runs top-3 retrieval against
   each index, and checks whether the correct chunk is in the top 3 —
   aggregated by fact_type.
4. `analyze_failures.py` diffs baseline vs. compressed retrieval per-question
   to find which specific facts regress under quantization (not just the
   aggregate %).

## Results (recall@3, n=104 questions)

| fact_type | baseline | turbovec_b4 | turbovec_b2 |
|-----------|----------|-------------|-------------|
| date      | 75.0%    | 83.3%       | 75.0%       |
| id        | 75.0%    | 75.0%       | 75.0%       |
| name      | 73.5%    | 76.5%       | 73.5%       |
| number    | 87.0%    | 84.8%       | 80.4%       |
| **overall** | **79.8%** | **80.8%** | **76.9%** |

## What this actually shows

- **`number` facts degrade monotonically with compression**: 87.0% → 84.8% →
  80.4% as bit_width drops from full precision to 4 to 2. It's also the
  largest category (n=46), so this trend is the most trustworthy signal in
  the dataset.
- **The real mechanism isn't "numbers are hard" — it's templated chunks
  colliding.** Digging into the actual regressions (`analyze_failures.py`)
  shows 3 of 5 baseline→b2 failures are all `RACK-XX` server-spec questions
  (CPU core counts). Every server-spec chunk uses the identical sentence
  template — "Rack RACK-XX at LOCATION runs N CPU cores and M GB RAM..." —
  so they already sit close together in embedding space; the number is the
  *only* thing distinguishing them. Quantization noise is enough to erase
  that thin margin, and the chunks get pushed into each other — C017, C019,
  C020, C021, C022 start swapping places in the top-3. A number embedded in
  a more varied, less template-like sentence (like a financial summary) has
  more surrounding context to anchor on and holds up better. So the risk
  isn't "numeric data" as a category — it's **near-duplicate/templated
  chunks whose only differentiator is a number**, which is exactly the
  shape of a lot of real infra/log data (repeated record formats with one
  changing field).
- **`id`, `date`, and `name` are noisy but roughly flat** at this sample
  size (n=12 for id/date) — a 1-question swing is an 8-point jump, so no
  confident claim there without a bigger ground-truth set.
- **bit_width=4 looks "free"** — overall recall is even marginally *above*
  baseline, most likely noise from such a small chunk count (22), not real
  improvement from compression.
- **The more surprising finding: baseline itself only hits 79.8% recall@3.**
  Semantic embedding search is mediocre at exact-fact retrieval even with
  zero compression — 19 of 104 questions miss on baseline too, mostly
  proper-noun/ID lookups (ticket IDs, engineer names) that don't have much
  semantic pull to begin with. TurboVec isn't the only bottleneck here;
  quantization makes an existing, secondary problem worse rather than being
  the root cause.

## When compression is fine — and when it isn't

The rule that falls out of this test: **quantization is safe where meaning
and vibe dominate retrieval, and unsafe where a unique, deterministic fact
is the only thing that matters.** Aggressive compression doesn't destroy
semantic similarity — sentiment, tone, topic all survive 16x compression
fine. What it destroys is the thin, precise signal that distinguishes one
near-identical record from another.

### 🟢 Green zone — aggressive (2–4 bit) quantization is fine here

- **Support FAQs / general how-tos** — "how do I reset my password" only
  needs the right *neighborhood* of meaning, not millimeter precision.
- **Marketing copy / brand voice** — searching for "posts that were funny"
  has no exact answer to get wrong.
- **Sentiment / review analysis** — happy vs. angry is a strong signal that
  survives heavy compression noise easily.
- **Creative brainstorming** — there's no ground truth to corrupt; a little
  vector noise can even surface more varied associations.
- **Internal wiki / soft HR content** — narrative text with no IDs or fixed
  numbers to get wrong.

### 🔴 Red zone — don't compress here

- **Financial & accounting data** — transaction IDs, exact amounts, IBANs.
  A missed decimal or a swapped transaction is a trust-destroying failure,
  not a rounding error.
- **Legal & compliance text** — one word ("may" vs. "must") changes the
  entire clause's meaning; "we used 2-bit compression" isn't a defense in
  court.
- **Technical logs / error codes** — this is what quantcheck actually
  measured: templated, near-identical records (server specs, incident
  logs) where the only differentiator is a number are exactly where
  bit_width=2 showed a real, measurable regression in this test.
- **Security & auth logs** — reconstructing the exact sequence of who/when/
  from-where after an incident needs bit-for-bit fidelity, not "semantically
  similar."
- **Medical/health data** — 10mg and 100mg look close in vector space and
  are not close in the real world.

The takeaway isn't "numbers are bad" in the abstract — it's that **anywhere
your data is templated, near-duplicate, and distinguished only by a
number or ID**, that's the specific shape of data to stress-test before
trusting aggressive compression, regardless of the domain label.

## A note on scale

This test runs on 22 chunks — a 4-point recall drop there is 1-2 missed
questions, easy to shrug off. It does not stay easy to shrug off at
production scale. The percentage doesn't change with corpus size, but the
absolute count of affected records does: the same 4% drop across, say,
300,000 financial records is roughly **12,000 records dropping out of
top-3 retrieval** — each one a complete information gap, not a slightly-off
number. A miss means the LLM never sees that chunk at all; it either
hallucinates a number or says nothing.

So treat the recall percentages in this README as a *direction and
mechanism*, not a magnitude you can extrapolate directly — this test tells
you templated near-duplicate chunks are the risk shape and roughly how much
worse b2 gets vs. b4, but the real-world cost of that percentage scales with
how many records you actually have, not with how small this test set is.

## Practical takeaway

If a RAG pipeline needs to retrieve precise numeric facts reliably,
TurboVec at bit_width=2 is where I'd stop trusting it — the `number`
category shows a clear, consistent drop by that point, concentrated in
templated records where a number is the only distinguishing feature
(server specs, log rows, repeated-format records). bit_width=4 seems safe.
If your corpus is full of near-duplicate templated chunks (which most
infra/log data is), that's the specific case to stress-test before trusting
aggressive compression — not "numbers" in general. And regardless of
compression level, semantic embedding retrieval alone is the wrong tool for
exact-fact lookup; a production system should pair it with structured
extraction or keyword/exact-match retrieval for numeric and ID fields.

## Reproducing

**Files you run, in this order:**

```bash
# 1. Generate the synthetic report + ground truth (no arguments needed)
python fake_report.py

# 2. Build the three indexes — point at your local sentence-transformers
#    snapshot, either way works:
python embed_and_index.py --model-path /path/to/all-MiniLM-L6-v2/snapshot
#   or, set it once for the session and skip the flag from here on:
#   export QUANTCHECK_MODEL_PATH=/path/to/all-MiniLM-L6-v2/snapshot   (Windows: set QUANTCHECK_MODEL_PATH=...)

# 3. Run all 104 questions against all three indexes, get the results table
python query_test.py

# 4. Optional — dig into specific failures instead of just the aggregate %
python analyze_failures.py --fact-type number --index turbovec_b2
python analyze_by_section.py
```

**Files you do NOT run directly:**

- `chunk_id_codec.py` — this is a helper module, not a script. It has no
  `main()` and does nothing if you run it. `embed_and_index.py` and
  `query_test.py` both `import` from it internally to convert chunk_ids
  (`"C001"`) to the `uint64` ids TurboVec's `IdMapIndex` requires, and back.
  You never call it yourself.

`analyze_failures.py` prints the exact questions where a compressed index
missed a fact the baseline caught — useful for spot-checking *which* numbers
break, not just the aggregate rate. `analyze_by_section.py` checks whether
a specific section (e.g. financial data) is disproportionately affected,
rather than relying on the fact_type aggregate alone.

Any local sentence-transformers snapshot works, not just all-MiniLM-L6-v2 —
the model just needs to be loadable via
`SentenceTransformer(path, local_files_only=True)`. Swap in your own
`report.txt` and `ground_truth.json` to test against your own data instead
of the synthetic report included here. Your `report.txt` needs to use the
same chunk-delimiter format `embed_and_index.py` parses — each chunk starts
with a `[[chunk:CXXX]]` marker on its own line, followed by the chunk text:

```
[[chunk:C001]]
Incident Report #1 (Ticket INC-98696)
On 2026-04-08, server cluster Helsinki-DC3 experienced a cold cache
stampede event...

[[chunk:C002]]
Financial Summary — Q1 2026
Total revenue for Q1 2026 was EUR 412,003.50...
```

Chunk ids must be `C` followed by digits (`C001`, `C022`, `C1234`, etc.) —
that's what `chunk_id_codec.py` converts to/from `uint64` for TurboVec.
