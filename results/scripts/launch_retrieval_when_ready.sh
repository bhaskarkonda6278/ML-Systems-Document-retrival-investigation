#!/bin/bash
set -euo pipefail
cd "$HOME/edin-mls-26-autumn"
while ! test -f results/setup/rag_setup.exit || ! test -f results/asr/model_checks.exit; do sleep 10; done
for file in results/setup/rag_setup.exit results/asr/model_checks.exit; do if test "$(cat "$file")" != 0; then printf 'Prerequisite failed: %s
' "$file"; exit 1; fi; done
sbatch --parsable --job-name=mls-rag-retrieval --output=results/rag/retrieval-job-%j.log results/scripts/rag_retrieval.sh > results/rag/retrieval_job.id
cat results/rag/retrieval_job.id
