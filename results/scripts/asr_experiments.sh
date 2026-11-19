#!/usr/bin/env bash
set -eo pipefail
ROOT="$HOME/edin-mls-26-autumn"
trap 'result=$?; printf "%s\n" "$result" > "$ROOT/results/asr/experiments.exit"' EXIT
export PATH=/opt/cuda-12.8.0/bin:$PATH
cd "$ROOT/hw1-asr"
python3 run_experiments.py
