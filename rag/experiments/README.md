# Investigation evidence archives

- baseline_first5/: previously archived initial baseline.
- adaptive_depth_study/: completed 30-tune/20-validation/50-dev retrieval study,
  fixed settings, per-query metrics, diagnostics, figures and separate judge audit.
- combined_gpu_api_check_v3/: successful two-question ELM integration check;
  integration evidence only, not a benchmark conclusion.
- combined_gpu_dev125_v3/: completed four-system, 125-dev, three-repeat evaluation.

Source is in rag/extensions/; study outputs are generated in rag/results/ on the
cluster. These experiment folders deliberately track selected reproducibility
artifacts without unignoring every generated result. They preserve the recorded
protocol/source hashes; do not relabel historical hashes as this documentation
commit. Large raw prompts/answers, logs, weights, assets, .env and backups are
excluded. See each archive's README and the investigation README for limitations.

To rebuild the retrieval analysis in a fresh clone, install course assets and
copy the adaptive_depth_study archive into an otherwise empty matching
rag/results/adaptive_depth_study directory. Existing cluster results should
remain untouched. The combined compact metrics allow independent comparisons,
but the runner's original analyse-only mode needs the cluster raw trial/judge files.
