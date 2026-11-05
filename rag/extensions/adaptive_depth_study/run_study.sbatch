#!/bin/bash
#SBATCH --job-name=rag-adaptive-study
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=rag/results/adaptive_depth_study/slurm-%j.log
set -euo pipefail
export PATH=/opt/cuda-12.8.0/bin:$PATH
export RAG_ASR_PYTHON=/home/s3021085/miniconda3/envs/mls-asr/bin/python
export ASR_CUDA=none
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
source /home/s3021085/mls-rag-env/bin/activate
python -u rag/extensions/adaptive_depth_study/study.py "$@"
