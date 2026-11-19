#!/usr/bin/env bash
#SBATCH -p Teaching
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -eo pipefail
ROOT="$HOME/edin-mls-26-autumn"
trap 'result=$?; printf "%s\n" "$result" > "$ROOT/results/asr/model_checks.exit"' EXIT
export PATH=/opt/cuda-12.8.0/bin:$PATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls-asr
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT/hw1-asr"
hostname | tee "$ROOT/results/asr/model_hardware.log"
nvidia-smi -L | tee -a "$ROOT/results/asr/model_hardware.log"
python -c 'import torch,transformers; print("torch",torch.__version__,"transformers",transformers.__version__)' | tee -a "$ROOT/results/asr/model_hardware.log"
make test | tee "$ROOT/results/asr/final_make_test.log"
./run.sh --cuda none --profile | tee "$ROOT/results/asr/pytorch_profile.log"
./run.sh --cuda task1 --warmup 0 --runs 1 | tee "$ROOT/results/asr/task1_model.log"
./run.sh --cuda task1,task2 --warmup 0 --runs 1 | tee "$ROOT/results/asr/task2_model.log"
./run.sh --cuda task1,task2,task3 --warmup 0 --runs 1 | tee "$ROOT/results/asr/task3_model.log"
./run.sh --profile | tee "$ROOT/results/asr/cuda_profile.log"
