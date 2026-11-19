#!/usr/bin/env bash
set -eo pipefail
ROOT="$HOME/edin-mls-26-autumn"
trap 'result=$?; printf "%s\n" "$result" > "$ROOT/results/setup/rag_setup.exit"' EXIT
cd "$ROOT"
git lfs install --local
git lfs pull --include="rag-assets.zip"
python3 -m venv "$HOME/mls-rag-env"
source "$HOME/mls-rag-env/bin/activate"
python -m pip install -r rag/requirements.txt
if ! test -f rag/.env; then cp rag/.env.example rag/.env; chmod 600 rag/.env; fi
mkdir -p "$HOME/mls-rag-assets"
python - <<'PY'
from pathlib import Path
import zipfile
archive = zipfile.ZipFile('rag-assets.zip')
target = (Path.home()/'mls-rag-assets').resolve()
for member in archive.infolist():
    path = (target/member.filename).resolve()
    if path != target and target not in path.parents:
        raise ValueError('Archive path outside asset directory')
archive.extractall(target)
PY
python rag/scripts/install_assets.py --from-dir "$HOME/mls-rag-assets"
python -m unittest discover -s rag/tests -v
