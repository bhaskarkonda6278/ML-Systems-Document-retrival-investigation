import unittest
from policies import adaptive_k, guarded_k


class PolicyTests(unittest.TestCase):
    def test_cutoff_rank_and_buffer(self):
        result = adaptive_k([.9, .89, .5, .49, .48, .47, .46, .45, .44, .43], buffer=2)
        self.assertEqual(result["raw_k"], 2)
        self.assertEqual(result["k"], 4)

    def test_excludes_large_tail_gap(self):
        scores = [1-i*.01 for i in range(9)] + [0]
        result = adaptive_k(scores, buffer=0, search_fraction=.9)
        self.assertLess(result["raw_k"], 9)

    def test_bounds_and_flat_fallback(self):
        result = adaptive_k([1, .5, .49, .48], buffer=5, max_k=3)
        self.assertEqual(result["k"], 3)
        self.assertTrue(result["clipped"])
        result = guarded_k([.5]*30, threshold=.1, fixed_k=5)
        self.assertEqual(result["k"], 5)
        self.assertTrue(result["fallback"])

    def test_invalid_values_and_single_candidate(self):
        for values in ([], [.5, float("nan")], [.4, .5]):
            with self.assertRaises(ValueError):
                adaptive_k(values)
        self.assertEqual(adaptive_k([.5])["k"], 1)

    def test_guard_can_select_below_baseline_without_labels(self):
        scores=[.95,.5]+[.49-i*.001 for i in range(28)]
        result=guarded_k(scores,threshold=.5,fixed_k=5,buffer=0)
        self.assertEqual(result["k"],1)
        self.assertFalse(result["fallback"])

    def test_positive_affine_score_transform_preserves_decision(self):
        scores=[.95,.5]+[.49-i*.001 for i in range(28)]
        a=guarded_k(scores,threshold=.5,fixed_k=5,buffer=0)
        b=guarded_k([3*s+7 for s in scores],threshold=.5,fixed_k=5,buffer=0)
        self.assertEqual(a["k"],b["k"])
        self.assertAlmostEqual(a["prominence"],b["prominence"])


if __name__ == "__main__":
    unittest.main()
