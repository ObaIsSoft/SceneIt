"""Exact cosine-similarity index over unit-length vectors, keyed by explicit ids.

Why this exists instead of FAISS
--------------------------------
Two reasons, one forced and one chosen.

*Forced:* faiss-cpu and torch each bundle their own copy of libomp. On macOS
arm64, importing both and then calling a faiss search aborts the process with
"OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
initialized". app.py imports both, so the app could not complete a search on a
development Mac at all. The documented escape hatch, KMP_DUPLICATE_LIB_OK=TRUE,
is described upstream as unsupported and liable to "silently produce incorrect
results" -- which is precisely the failure mode this refactor exists to remove.

*Chosen:* the corpus is a few thousand vectors. A brute-force matmul over
3,000 x 512 float32 is on the order of a millisecond, so approximate search was
buying nothing while costing a 19 MB dependency, an untested IVF code path and
an `nprobe` that nothing tuned. Approximate indexing earns its place somewhere
north of a hundred thousand vectors; at that point this class is the place to
swap in HNSW or a binary-then-rescore two-stage, behind the same interface.

The `search()` signature mirrors faiss deliberately -- `(scores, ids)` as
2-D arrays, short rows padded with -1 -- so callers do not care which backend
is underneath.
"""

import numpy as np

# Returned in place of a missing neighbour, matching faiss's convention.
EMPTY_ID = -1


class VectorIndex:
    def __init__(self, dim):
        self.dim = int(dim)
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._ids = np.zeros((0,), dtype=np.int64)

    # --- Introspection ----------------------------------------------------

    @property
    def ntotal(self):
        return int(self._ids.shape[0])

    @property
    def ids(self):
        return self._ids.copy()

    def __len__(self):
        return self.ntotal

    # --- Building ---------------------------------------------------------

    def add_with_ids(self, vectors, ids):
        """Add vectors under caller-supplied ids.

        Vectors are normalised defensively: cosine similarity is only equal to
        the inner product for unit-length input, and a single unnormalised row
        would otherwise outrank everything in the index.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.int64).ravel()

        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {vectors.shape[1]}")
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(f"got {vectors.shape[0]} vectors but {ids.shape[0]} ids")

        duplicate = np.intersect1d(ids, self._ids)
        if duplicate.size:
            raise ValueError(f"ids already present in the index: {duplicate[:5].tolist()}")
        if np.unique(ids).size != ids.size:
            raise ValueError("duplicate ids within the batch")

        self._vectors = np.vstack([self._vectors, _normalise(vectors)])
        self._ids = np.concatenate([self._ids, ids])

    def remove_ids(self, ids):
        """Drop vectors by id. Returns how many were removed."""
        ids = np.ascontiguousarray(ids, dtype=np.int64).ravel()
        keep = ~np.isin(self._ids, ids)
        removed = int((~keep).sum())
        self._vectors = self._vectors[keep]
        self._ids = self._ids[keep]
        return removed

    # --- Searching --------------------------------------------------------

    def search(self, queries, k):
        """Return (scores, ids), each shaped (n_queries, k).

        Scores are cosine similarities in [-1, 1], descending. Rows are padded
        with EMPTY_ID when the index holds fewer than k vectors.
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.shape[1] != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {queries.shape[1]}")

        n_queries = queries.shape[0]
        k = max(1, int(k))

        if self.ntotal == 0:
            return (
                np.full((n_queries, k), -1.0, dtype=np.float32),
                np.full((n_queries, k), EMPTY_ID, dtype=np.int64),
            )

        # Unit vectors on both sides, so the inner product is the cosine.
        sims = _normalise(queries) @ self._vectors.T

        take = min(k, self.ntotal)
        # argpartition for the top-k, then sort just those -- O(n) rather than
        # a full O(n log n) sort of the corpus per query.
        part = np.argpartition(-sims, take - 1, axis=1)[:, :take]
        ordered = np.take_along_axis(part, np.argsort(-np.take_along_axis(sims, part, 1), axis=1), 1)

        scores = np.take_along_axis(sims, ordered, axis=1).astype(np.float32)
        ids = self._ids[ordered]

        if take < k:
            pad = k - take
            scores = np.hstack([scores, np.full((n_queries, pad), -1.0, dtype=np.float32)])
            ids = np.hstack([ids, np.full((n_queries, pad), EMPTY_ID, dtype=np.int64)])

        return scores, ids

    def reconstruct(self, vector_id):
        """Return the stored vector for an id."""
        match = np.flatnonzero(self._ids == int(vector_id))
        if match.size == 0:
            raise KeyError(f"id {vector_id} is not in the index")
        return self._vectors[match[0]].copy()

    # --- Persistence ------------------------------------------------------

    def save(self, path):
        np.savez(path, vectors=self._vectors, ids=self._ids, dim=np.int64(self.dim))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            index = cls(int(data["dim"]))
            index._vectors = np.ascontiguousarray(data["vectors"], dtype=np.float32)
            index._ids = np.ascontiguousarray(data["ids"], dtype=np.int64)
        if index._vectors.shape[0] != index._ids.shape[0]:
            raise ValueError(
                f"corrupt index at {path}: {index._vectors.shape[0]} vectors, "
                f"{index._ids.shape[0]} ids"
            )
        return index


def _normalise(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Leave zero vectors alone rather than dividing by zero; they score 0
    # against everything, which is the correct outcome for an empty embedding.
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors
