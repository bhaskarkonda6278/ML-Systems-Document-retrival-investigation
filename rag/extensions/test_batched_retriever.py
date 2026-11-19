"""Small, manually checkable ranking and input-contract tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.batched_retriever import BatchedExactDenseRetriever


class BatchRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        np.save(root / "vectors.npy", np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32))
        (root / "ids.json").write_text(json.dumps(["east", "north", "west"]))
        self.retriever = BatchedExactDenseRetriever(root / "vectors.npy", root / "ids.json")

    def tearDown(self):
        self.directory.cleanup()

    def test_scaled_queries_rank_independently_without_mutation(self):
        queries = np.array([[4, 1], [-3, 1]], dtype=np.float32)
        original = queries.copy()
        results = self.retriever.retrieve_many(queries, k=3)["results"]
        self.assertEqual(results[0]["record_ids"], ["east", "north", "west"])
        self.assertEqual(results[1]["record_ids"], ["west", "north", "east"])
        np.testing.assert_array_equal(queries, original)
        self.assertAlmostEqual(results[0]["scores"][0], 4 / np.sqrt(17), places=6)

    def test_empty_batch_and_k_clamping(self):
        self.assertEqual(self.retriever.retrieve_many(np.empty((0, 2)))["results"], [])
        self.assertEqual(len(self.retriever.retrieve_many([[2, 1]], k=99)["results"][0]["record_ids"]), 3)

    def test_invalid_queries_are_rejected(self):
        for query in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[1, 2, 3]], [1, 2]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.retriever.retrieve_many(query)
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.retriever.retrieve_many([[1, 2]], k=k)


if __name__ == "__main__":
    unittest.main()
