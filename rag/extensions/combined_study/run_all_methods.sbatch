#!/bin/bash
#SBATCH -p Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=combined_rag
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=/opt/cuda-12.8.0/bin:$PATH
export RAG_ASR_PYTHON="$HOME/miniconda3/envs/mls-asr/bin/python"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$HOME/mls-rag-env/bin/activate"
hostname
nvidia-smi -L
python -u rag/extensions/combined_study/run_all_methods.py "$@"
