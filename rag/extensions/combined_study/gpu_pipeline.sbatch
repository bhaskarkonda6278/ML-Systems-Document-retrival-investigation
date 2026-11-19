#!/bin/bash
#SBATCH -p Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=combined_gpu
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=/opt/cuda-12.8.0/bin:$PATH
export RAG_ASR_PYTHON="$HOME/miniconda3/envs/mls-asr/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$HOME/mls-rag-env/bin/activate"
hostname
nvidia-smi -L

if [ "$1" = smoke ]; then
    python -u rag/extensions/combined_study/run.py \
        --split practice --limit 2 --repeats 1 \
        --output rag/results/combined_gpu_api_check
    python rag/extensions/combined_study/validate_smoke.py \
        rag/results/combined_gpu_api_check
else
    python rag/extensions/combined_study/validate_smoke.py \
        rag/results/combined_gpu_api_check
    python -u rag/extensions/combined_study/run.py \
        --split dev --repeats 3 \
        --output rag/results/combined_gpu_dev125
fi
