#!/usr/bin/env bash
# Run on an allocated GPU node after the user's ELM key has been configured.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PATH=/opt/cuda-12.8.0/bin:$PATH
export RAG_ASR_PYTHON="$HOME/miniconda3/envs/mls-asr/bin/python"
source "$HOME/mls-rag-env/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python rag/scripts/check_setup.py
python - <<'PY'
from pathlib import Path
import os
from dotenv import dotenv_values
settings = dotenv_values(Path('rag/.env'))
if not (os.environ.get('ELM_API_KEY') or settings.get('ELM_API_KEY')):
    raise SystemExit('A genuine ELM_API_KEY is required; no model calls have been made.')
PY
make -C hw1-asr lib
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
out="results/elm/$run_id"
mkdir -p "$out"
ASR_CUDA=none python rag/scripts/run_audio_query.py --device cpu | tee "$out/reference_query.log"
ASR_CUDA=all python rag/scripts/run_audio_query.py --device cpu | tee "$out/custom_query.log"
export ASR_CUDA=all
python -u rag/scripts/generate_answers.py --device cpu --split practice --limit 5 --output "$out/sample_answers.jsonl"
python rag/scripts/evaluate_answers.py --split practice --candidates "$out/sample_answers.jsonl" --allow-subset --dry-run
python rag/scripts/evaluate_answers.py --split practice --candidates "$out/sample_answers.jsonl" --allow-subset --repeats 3 --out-dir "$out/sample_judge"
printf 'Diagnostic outputs: %s\n' "$out"
printf 'Inspect evidence, answers and judge results before a full-split run.\n'
