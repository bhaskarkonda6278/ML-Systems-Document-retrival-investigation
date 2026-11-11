"""Offline checks for the complete ASR/retrieval matrix and depth controls."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'expanded_runner', Path(__file__).with_name('run_all_methods.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AllMethodsTests(unittest.TestCase):
    def test_complete_matrix(self):
        policies = {'fixed1', 'fixed3', 'fixed5', 'fixed10', 'fixed20',
                    'gap0', 'buffer5'}
        expected = {(mode, policy) for mode in ['baseline', 'kv_trim']
                    for policy in policies}
        self.assertEqual(len(runner.SYSTEMS), 14)
        self.assertEqual(set(runner.SYSTEMS.values()), expected)

    def test_fixed_counts(self):
        for k in [1, 3, 5, 10, 20]:
            self.assertEqual(runner.retrieval_count(f'fixed{k}'), k)
            self.assertEqual(runner.choose_depth(f'fixed{k}', [1.0]*100)['k'], k)

    def test_buffer_and_candidate_pool(self):
        scores = [1.0] + [0.5-i*0.001 for i in range(99)]
        self.assertEqual(runner.choose_depth('gap0', scores)['k'], 1)
        self.assertEqual(runner.choose_depth('buffer5', scores)['k'], 6)
        self.assertEqual(runner.retrieval_count('gap0'), 100)
        self.assertEqual(runner.retrieval_count('buffer5'), 100)
        with self.assertRaises(ValueError):
            runner.choose_depth('unknown', scores)


if __name__ == '__main__':
    unittest.main()
