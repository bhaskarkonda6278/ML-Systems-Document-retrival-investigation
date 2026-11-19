#!/usr/bin/env bash
#SBATCH -p Teaching
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -eo pipefail
ROOT="$HOME/edin-mls-26-autumn"
trap 'result=$?; printf "%s\n" "$result" > "$ROOT/results/rag/retrieval_checks.exit"' EXIT
export PATH=/opt/cuda-12.8.0/bin:$PATH
export RAG_ASR_PYTHON="$HOME/miniconda3/envs/mls-asr/bin/python"
source "$HOME/mls-rag-env/bin/activate"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ASR_CUDA=none
cd "$ROOT"
hostname | tee results/rag/hardware.log
nvidia-smi -L | tee -a results/rag/hardware.log
make -C hw1-asr lib
python rag/scripts/check_setup.py > results/rag/setup_check.log 2>&1 || setup_status=$?
printf '%s\n' "${setup_status:-0}" > results/rag/setup_check.exit
python -m unittest discover -s rag/tests -v
python -m unittest discover -s rag/extensions -p 'test_batched_retriever.py' -v
python rag/extensions/inspect_audio_retrieval.py --cuda none --output results/rag/sample_reference.json
python rag/extensions/inspect_audio_retrieval.py --cuda all --output results/rag/sample_custom_cuda.json
python -u rag/scripts/evaluate_rag.py --device cpu --split practice --transcript-cache rag/.cache/practice_transcripts.jsonl | tee results/rag/practice_retrieval.log
python -u rag/scripts/evaluate_rag.py --device cpu --split dev --transcript-cache rag/.cache/dev_transcripts.jsonl | tee results/rag/dev_retrieval.log
python -u rag/extensions/benchmark_batching.py --split practice --transcript-cache rag/.cache/practice_transcripts.jsonl --out-dir results/rag/batching_practice | tee results/rag/batching_practice.log
python -u rag/extensions/benchmark_batching.py --split dev --transcript-cache rag/.cache/dev_transcripts.jsonl --out-dir results/rag/batching_dev | tee results/rag/batching_dev.log
