"""
chunk_id_codec.py

turbovec.IdMapIndex requires external ids as uint64. Our chunk_ids look like
"C001", "C022", etc. (from fake_report.py's [[chunk:CXXX]] markers). This is
the single place that converts between the two, so embed_and_index.py and
query_test.py can't drift out of sync.
"""

import numpy as np


def chunk_id_to_uint64(chunk_id):
    """'C001' -> np.uint64(1)"""
    return np.uint64(int(chunk_id[1:]))


def chunk_ids_to_uint64_array(chunk_ids):
    return np.array([chunk_id_to_uint64(c) for c in chunk_ids], dtype=np.uint64)


def uint64_to_chunk_id(value):
    """np.uint64(1) -> 'C001'"""
    return f"C{int(value):03d}"
