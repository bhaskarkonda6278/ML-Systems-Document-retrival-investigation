# Reproducing the coursework results

Base: https://github.com/ed-aisys/edin-mls-26-autumn, revision `d0b99ee`.
Use a fresh checkout for a new measurement run so existing evidence is preserved.
Apply the five submitted CUDA files and copy the supplied extension and experiment
scripts into the same relative paths.

## Environment and data

Run `bash hw1-asr/setup_env.sh` on the head node, as the course guide specifies.
Keep ASR in `mls-asr`, and create the separate RAG environment using the course
README. The exact installed package versions from the recorded runs are in
`results/setup/asr_requirements_frozen.txt` and `rag_requirements_frozen.txt`.
The model snapshot used was `61ba4e0b3309b6656edea3e93e419f7bd5c61957`.

Install the course data with Git LFS and the provided asset installer:

```bash
git lfs install --local
git lfs pull --include="rag-assets.zip"
mkdir -p "$HOME/mls-rag-assets"
unzip rag-assets.zip -d "$HOME/mls-rag-assets"
source "$HOME/mls-rag-env/bin/activate"
python rag/scripts/install_assets.py --from-dir "$HOME/mls-rag-assets"
```

The archive contains 24,000 corpus records, 350 practice queries, 125 dev queries,
public relevance judgments, enc_a/enc_b corpus vectors and the enc_a query model.
Data and model weights are not part of the submission package.

## ASR correctness and profiling

On an allocated GPU node:

```bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls-asr
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd hw1-asr
nvidia-smi -L
make test
./run.sh --cuda none --profile
./run.sh --cuda task1 --warmup 0 --runs 1
./run.sh --cuda task1,task2 --warmup 0 --runs 1
./run.sh --cuda task1,task2,task3 --warmup 0 --runs 1
./run.sh --profile
```

The recorded comparison used an H200 MIG 1g.18gb slice (16 SMs), two requested
CPU cores, 16 GB host memory and CUDA 12.8.93. The PyTorch reference and custom
profiles were collected in the same GPU allocation, along with `make test`.
Both transcribed the supplied recording correctly. The reported accuracy is the
benchmark's expected unique-word overlap, not a general WER estimate.

## Required kernel experiments

In the same kind of GPU allocation, from `hw1-asr/`:

```bash
python run_experiments.py
```

The driver copies sources into variant directories and leaves canonical task
files unchanged. It sweeps six RoPE block sizes, six reduction block sizes,
three GEMM tile widths and five attention warp counts. Each configuration has
three test executions. It also runs small-case memcheck/racecheck. The supplied
Makefile and tests are unchanged. Full data are in
`results/experiments/experiments.json` and `measurements.csv`.

## RAG retrieval and batching investigation

From the repository root, on a GPU node:

```bash
export RAG_ASR_PYTHON="$HOME/miniconda3/envs/mls-asr/bin/python"
source "$HOME/mls-rag-env/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export ASR_CUDA=none HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m unittest discover -s rag/tests -v
python -m unittest discover -s rag/extensions -p test_batched_retriever.py -v
python rag/scripts/evaluate_rag.py --device cpu --split practice --transcript-cache rag/.cache/practice_transcripts.jsonl
python rag/scripts/evaluate_rag.py --device cpu --split dev --transcript-cache rag/.cache/dev_transcripts.jsonl
python rag/extensions/benchmark_batching.py --split practice --transcript-cache rag/.cache/practice_transcripts.jsonl --out-dir results/rag/batching_practice
python rag/extensions/benchmark_batching.py --split dev --transcript-cache rag/.cache/dev_transcripts.jsonl --out-dir results/rag/batching_dev
```

The extension freezes actual transcripts and single-query enc_a embeddings,
then changes search batching only. It compares k=100 exact search at batch sizes
1, 4, 16, 64 and 128, after warm-up, over seven rounds in randomized order.
The CPU model and actual BLAS thread pools are recorded with the results.
Quality is measured independently from latency and extra traced allocations.
Encoding, ASR, online waiting and ELM generation are excluded from search timing.
Use a new transcript cache if comparing different ASR implementations.

## ELM work still requiring a genuine key

The user's ELM key request is prepared but has not been submitted. Once the key
is issued, store it only in `rag/.env` (excluded from Git, mode 600) or an
environment variable. Do not put it in code, logs, the report or this package.
Run `rag/extensions/complete_elm_evaluation.sh` on a GPU node. It checks setup,
runs one complete query with each ASR backend, generates five diagnostic answers,
validates candidate coverage, and judges the answers three times using the
supplied evaluator. Inspect evidence and outputs before a broader run. Any
subset results must remain labeled as such. No answer results are fabricated.

## Report

Install `report/requirements.txt` into a report environment, then run:

```bash
python report/build_report.py
cd report/latex_template
pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
cd ../..
python report/format_checker/check_report_format.py report/latex_template/report.pdf
```

The original style, bibliography style, review mode, author format and course
header are preserved. The main body must remain within six pages. Complete the
group-information page and review the accurate AI-use declaration before any
submission. The report remains a draft until the pending ELM work and group
information are resolved.
