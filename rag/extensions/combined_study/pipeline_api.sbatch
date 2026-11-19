#!/bin/bash
#SBATCH -p Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=combined_pipeline
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
 python rag/extensions/combined_study/validate_smoke.py rag/results/combined_smoke_ready
 python -u rag/extensions/combined_study/run.py --split practice --limit 2 --repeats 1 --output rag/results/combined_smoke_api
 python rag/extensions/combined_study/validate_smoke.py rag/results/combined_smoke_api
elif [ "$1" = full ]; then
 python rag/extensions/combined_study/validate_smoke.py rag/results/combined_smoke_api
 python -u rag/extensions/combined_study/run.py --split dev --repeats 3 --output rag/results/combined_dev125
else
 echo 'Expected smoke or full' >&2
 exit 2
fi
