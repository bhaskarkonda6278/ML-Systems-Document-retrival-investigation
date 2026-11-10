"""Offline checks for policy coverage and query-weighted paired aggregation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('combined_runner', HERE/'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CombinedTests(unittest.TestCase):
    def test_factorial_covers_each_pair_once(self):
        self.assertEqual(set(runner.SYSTEMS.values()), {
            ('baseline','fixed5'),('kv_trim','fixed5'),
            ('baseline','gap0'),('kv_trim','gap0')})

    def test_query_weighting_and_unseen_subgroup(self):
        trials, judgments = [], []
        measurements = ['asr_ms','asr_inference_ms','embedding_ms','retrieval_ms',
            'selection_ms','context_ms','generation_ms','end_to_end_ms',
            'prompt_tokens','completion_tokens','k','recall','peak_gpu_gib']
        for qid, seen, repeats in [('q1',True,3),('q2',False,1)]:
            for system in runner.SYSTEMS:
                for repeat in range(repeats):
                    key=f'{qid}:{system}:{repeat}'
                    value = 1.0 if qid=='q1' else 0.0
                    trials.append(dict(query_id=qid,system=system,judge_key=key,
                        seen_in_previous_study=seen,api_retries=0,
                        **{name:1 for name in measurements}))
                    judgments.append(dict(judge_key=key,score=value))
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for name, values in [('trials.jsonl',trials),('judgments.jsonl',judgments)]:
                (out/name).write_text(''.join(json.dumps(r)+'\n' for r in values))
            runner.analyse(out)
            report=json.loads((out/'analysis.json').read_text())
            self.assertEqual(report['all_dev']['B0']['mean_score'],0.5)
            self.assertEqual(report['previously_unseen75']['B0']['queries'],1)
            self.assertEqual(report['previously_unseen75']['B0']['mean_score'],0)
            for stats in report['all_dev'].values():
                self.assertEqual(stats['paired_query_ci95'],[0,0])


if __name__ == '__main__':
    unittest.main()
