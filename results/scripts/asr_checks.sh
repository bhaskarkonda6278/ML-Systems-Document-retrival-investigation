#!/usr/bin/env bash
set -eo pipefail
ROOT="$HOME/edin-mls-26-autumn"
trap 'result=$?; printf "%s\n" "$result" > "$ROOT/results/asr/kernel_tests.exit"' EXIT
export PATH=/opt/cuda-12.8.0/bin:$PATH
cd "$ROOT/hw1-asr"
hostname
nvidia-smi -L
nvcc --version
make -j2
make test
