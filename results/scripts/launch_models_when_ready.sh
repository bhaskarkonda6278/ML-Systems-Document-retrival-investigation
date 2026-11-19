#!/bin/bash
set -euo pipefail
cd "$HOME/edin-mls-26-autumn"
while ! test -f results/setup/asr_setup.exit; do sleep 10; done
if test "$(cat results/setup/asr_setup.exit)" != 0; then printf 'ASR setup failed
'; exit 1; fi
sbatch --parsable --job-name=mls-asr-validation --output=results/asr/model-job-%j.log results/scripts/asr_models.sh > results/asr/model_job.id
cat results/asr/model_job.id
