"""
embed_and_index.py

Reads data/report.txt (chunk-delimited by fake_report.py), embeds each chunk
with a LOCAL sentence-transformers snapshot (no Hugging Face network calls),
and builds two indexes:

  1. indexes/baseline.npz   - full-precision embeddings
  2. indexes/turbovec_b4.npz - TurboVec-compressed, bit_width=4
  3. indexes/turbovec_b2.npz - TurboVec-compressed, bit_width=2

All three keep the same chunk_id <-> row mapping so query_test.py can
compare which chunk each index retrieves for a given question.

IMPORTANT: point this at your own local sentence-transformers snapshot —
via --model-path, or the QUANTCHECK_MODEL_PATH env var. We force offline
mode so this never reaches out to Hugging Face.
"""

import argparse
import os

# --- force fully offline before importing anything HF-related -------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import re
import numpy as np
from sentence_transformers import SentenceTransformer

from chunk_id_codec import chunk_ids_to_uint64_array

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# No hardcoded path — set via --model-path or QUANTCHECK_MODEL_PATH so
# anyone can run this against their own local snapshot.
DEFAULT_MODEL_PATH = os.environ.get("QUANTCHECK_MODEL_PATH")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indexes")
REPORT_PATH = os.path.join(DATA_DIR, "report.txt")

CHUNK_PATTERN = re.compile(r"\[\[chunk:(C\d+)\]\]\n(.*?)(?=\n\[\[chunk:|\Z)", re.DOTALL)


def parse_chunks(path):
    """Split report.txt into (chunk_id, text) pairs using the [[chunk:CXXX]] markers."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    matches = CHUNK_PATTERN.findall(content)
    chunks = [{"chunk_id": cid, "text": text.strip()} for cid, text in matches]

    if not chunks:
        raise ValueError(f"No [[chunk:...]] markers found in {path}")

    return chunks


def embed_chunks(chunks, model):
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )
    return vectors.astype(np.float32)


def save_baseline(chunk_ids, vectors, out_path):
    np.savez(out_path, chunk_ids=np.array(chunk_ids), vectors=vectors)
    print(f"Wrote {out_path} — {vectors.shape[0]} vectors x {vectors.shape[1]} dims")


def save_turbovec(chunk_ids, vectors, bit_width, out_path):
    """
    Build a turbovec.IdMapIndex at the given bit_width and write it to disk
    as a .tvim file. External ids are derived from chunk_id via
    chunk_id_codec (e.g. "C001" -> 1), so query_test.py can map search
    results straight back to chunk_ids.
    """
    import turbovec

    ids = chunk_ids_to_uint64_array(chunk_ids)

    index = turbovec.IdMapIndex(dim=vectors.shape[1], bit_width=bit_width)
    index.add_with_ids(vectors, ids)
    index.prepare()
    index.write(out_path)

    print(f"Wrote {out_path} — bit_width={bit_width}, "
          f"{len(index)} vectors indexed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to a local sentence-transformers snapshot folder "
             "(config.json, model.safetensors, etc.). Falls back to the "
             "QUANTCHECK_MODEL_PATH env var if not given.",
    )
    return parser.parse_args()


# Resolved at import time so query_test.py can reuse the same default via
# `from embed_and_index import MODEL_PATH`. Overridden per-run by --model-path
# when this script is executed directly.
MODEL_PATH = DEFAULT_MODEL_PATH


def main():
    args = parse_args()
    model_path = args.model_path

    if not model_path:
        raise SystemExit(
            "No model path given. Pass --model-path /path/to/snapshot, or "
            "set the QUANTCHECK_MODEL_PATH environment variable."
        )
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}\n"
            "Check the snapshot folder is present and the hash hasn't changed."
        )

    os.makedirs(INDEX_DIR, exist_ok=True)

    print(f"Loading model from local snapshot: {model_path}")
    # local_files_only=True is the hard guarantee — even if an env var gets
    # dropped somewhere upstream, this forces sentence-transformers to never
    # touch the network.
    model = SentenceTransformer(model_path, local_files_only=True)

    print(f"Parsing chunks from {REPORT_PATH}")
    chunks = parse_chunks(REPORT_PATH)
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"Found {len(chunks)} chunks")

    print("Embedding chunks...")
    vectors = embed_chunks(chunks, model)

    save_baseline(chunk_ids, vectors, os.path.join(INDEX_DIR, "baseline.npz"))
    save_turbovec(chunk_ids, vectors, 4, os.path.join(INDEX_DIR, "turbovec_b4.tvim"))
    save_turbovec(chunk_ids, vectors, 2, os.path.join(INDEX_DIR, "turbovec_b2.tvim"))

    print("Done.")


if __name__ == "__main__":
    main()
