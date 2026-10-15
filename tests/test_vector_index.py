"""Unit tests for the replacement vector index.

Imports numpy only, never torch, so this file is safe from the libomp clash
that forced faiss out in the first place.
"""

import numpy as np
import pytest

from vector_index import EMPTY_ID, VectorIndex

pytestmark = pytest.mark.unit

DIM = 16


def unit(n, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestEmptyIndex:
    def test_reports_zero(self):
        assert VectorIndex(DIM).ntotal == 0
        assert len(VectorIndex(DIM)) == 0

    def test_search_returns_sentinels_not_an_error(self):
        """A query against an empty index must be answerable. app.py searches
        before any ingest has happened on a fresh install."""
        scores, ids = VectorIndex(DIM).search(unit(1), 5)
        assert ids.shape == (1, 5)
        assert (ids == EMPTY_ID).all()
        assert (scores < 0).all()


class TestAddAndSearch:
    def test_self_retrieval_scores_one(self):
        v = unit(10)
        idx = VectorIndex(DIM)
        idx.add_with_ids(v, np.arange(10))

        scores, ids = idx.search(v, 1)
        assert list(ids[:, 0]) == list(range(10))
        assert scores[:, 0] == pytest.approx(1.0, abs=1e-5)

    def test_results_are_descending(self):
        idx = VectorIndex(DIM)
        idx.add_with_ids(unit(50), np.arange(50))
        scores, _ = idx.search(unit(3, seed=9), 10)
        for row in scores:
            assert list(row) == sorted(row, reverse=True)

    def test_k_larger_than_corpus_is_padded(self):
        idx = VectorIndex(DIM)
        idx.add_with_ids(unit(3), np.arange(3))
        scores, ids = idx.search(unit(1, seed=4), 10)

        assert ids.shape == (1, 10)
        assert (ids[0][:3] != EMPTY_ID).all()
        assert (ids[0][3:] == EMPTY_ID).all()
        assert (scores[0][3:] < 0).all()

    def test_batch_queries(self):
        idx = VectorIndex(DIM)
        idx.add_with_ids(unit(20), np.arange(20))
        scores, ids = idx.search(unit(7, seed=3), 4)
        assert scores.shape == (7, 4)
        assert ids.shape == (7, 4)

    def test_one_dimensional_query_is_accepted(self):
        idx = VectorIndex(DIM)
        idx.add_with_ids(unit(5), np.arange(5))
        scores, ids = idx.search(unit(1)[0], 2)
        assert ids.shape == (1, 2)

    def test_unnormalised_input_is_normalised(self):
        """Cosine equals the inner product only for unit vectors. An
        unnormalised row would otherwise dominate every ranking."""
        v = unit(4)
        scaled = v.copy()
        scaled[2] *= 50.0

        idx = VectorIndex(DIM)
        idx.add_with_ids(scaled, np.arange(4))

        scores, _ = idx.search(v, 1)
        assert (scores[:, 0] <= 1.0001).all()
        # Row 2 still retrieves itself, at a similarity of 1 rather than 50.
        assert scores[2][0] == pytest.approx(1.0, abs=1e-5)


class TestIdSemantics:
    def test_sparse_ids_round_trip(self):
        ids = np.array([5, 900, 17, 3], dtype="int64")
        v = unit(4)
        idx = VectorIndex(DIM)
        idx.add_with_ids(v, ids)

        for position, wanted in enumerate(ids):
            _, got = idx.search(v[position : position + 1], 1)
            assert int(got[0][0]) == int(wanted)

    def test_duplicate_id_across_batches_is_refused(self):
        idx = VectorIndex(DIM)
        idx.add_with_ids(unit(2), [1, 2])
        with pytest.raises(ValueError, match="already present"):
            idx.add_with_ids(unit(1, seed=5), [2])

    def test_duplicate_id_within_a_batch_is_refused(self):
        idx = VectorIndex(DIM)
        with pytest.raises(ValueError, match="duplicate ids"):
            idx.add_with_ids(unit(2), [4, 4])

    def test_removal(self):
        idx = VectorIndex(DIM)
        idx.add_with_ids(unit(5), np.arange(5))
        assert idx.remove_ids([1, 3]) == 2
        assert idx.ntotal == 3
        assert sorted(idx.ids) == [0, 2, 4]

    def test_removal_does_not_shift_surviving_ids(self):
        """The whole point of explicit ids: deleting one must not renumber the
        others, which is exactly what positional mapping got wrong."""
        v = unit(5)
        idx = VectorIndex(DIM)
        idx.add_with_ids(v, np.arange(5))
        idx.remove_ids([2])

        _, ids = idx.search(v[4:5], 1)
        assert int(ids[0][0]) == 4


class TestValidation:
    def test_dimension_mismatch_on_add(self):
        idx = VectorIndex(DIM)
        with pytest.raises(ValueError, match="expected dimension"):
            idx.add_with_ids(unit(2, dim=DIM + 1), [0, 1])

    def test_dimension_mismatch_on_search(self):
        idx = VectorIndex(DIM)
        idx.add_with_ids(unit(2), [0, 1])
        with pytest.raises(ValueError, match="expected dimension"):
            idx.search(unit(1, dim=DIM + 3), 1)

    def test_count_mismatch(self):
        idx = VectorIndex(DIM)
        with pytest.raises(ValueError, match="but 1 ids"):
            idx.add_with_ids(unit(3), [0])

    def test_zero_vector_does_not_divide_by_zero(self):
        idx = VectorIndex(DIM)
        v = np.zeros((1, DIM), dtype="float32")
        idx.add_with_ids(v, [0])
        scores, ids = idx.search(unit(1), 1)
        assert int(ids[0][0]) == 0
        assert scores[0][0] == pytest.approx(0.0, abs=1e-6)


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        v = unit(12)
        ids = np.arange(100, 112, dtype="int64")
        original = VectorIndex(DIM)
        original.add_with_ids(v, ids)

        path = tmp_path / "idx.npz"
        original.save(str(path))
        restored = VectorIndex.load(str(path))

        assert restored.ntotal == original.ntotal
        assert restored.dim == original.dim
        assert sorted(restored.ids) == sorted(original.ids)

        for position, wanted in enumerate(ids):
            _, got = restored.search(v[position : position + 1], 1)
            assert int(got[0][0]) == int(wanted)

    def test_corrupt_archive_is_rejected(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(
            path,
            vectors=np.zeros((5, DIM), dtype="float32"),
            ids=np.arange(3, dtype="int64"),
            dim=np.int64(DIM),
        )
        with pytest.raises(ValueError, match="corrupt index"):
            VectorIndex.load(str(path))


class TestScale:
    @pytest.mark.slow
    def test_corpus_sized_search_is_fast(self):
        """3,000 vectors is the real corpus size. Exact search over it should be
        well inside a millisecond, which is the premise for dropping ANN."""
        import time

        idx = VectorIndex(512)
        idx.add_with_ids(unit(3000, dim=512, seed=1), np.arange(3000))
        query = unit(1, dim=512, seed=2)

        idx.search(query, 12)  # warm up
        start = time.perf_counter()
        for _ in range(20):
            idx.search(query, 12)
        per_query_ms = (time.perf_counter() - start) / 20 * 1000

        assert per_query_ms < 25, f"{per_query_ms:.2f} ms per query"
