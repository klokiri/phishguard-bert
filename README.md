# PhishGuard AI — BERT edition

Final-year capstone by Kasim: an explainable phishing detector for emails and URLs, with a BERT model, classical baselines, a Flask backend and the PhishGuard AI web front end.

> Work in progress. See [PROGRESS.md](PROGRESS.md) for status and [AI_ASSISTANCE_LOG.md](AI_ASSISTANCE_LOG.md) for AI-assistance declarations. Setup and reproduction steps are added as each phase lands. No results have been measured yet.

## Setup (macOS, Apple Silicon)

Tested on an Apple M4 Pro (macOS 26.5.1). Requires [uv](https://docs.astral.sh/uv/) and Homebrew.

```bash
brew install libomp
```

```bash
uv python install 3.12
```

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

```bash
.venv/bin/python scripts/check_env.py
```

The check prints library versions, whether CUDA or Apple's MPS GPU backend is available, runs a small computation on the GPU, and reports which mixed-precision modes work. It exits with code 0 when a GPU works.

On Windows/Linux with an NVIDIA GPU: install PyTorch for your CUDA version from https://pytorch.org/get-started/locally/ first, then `pip install -r requirements.txt` (skip `libomp`).

## Data

```bash
.venv/bin/python -m src.data
```

Downloads `train.json` and `test.json` (~551 MB) from [ealvaradob/phishing-dataset](https://huggingface.co/datasets/ealvaradob/phishing-dataset) at revision `23b8c17f`, which is the split the published model `ealvaradob/bert-finetuned-phishing` was trained and tested on. It verifies their SHA-256 hashes, takes a stratified 10% validation set from the training data (seed 42), and writes `data/splits/{train,val,test}.parquet`. It also writes the committed evidence files `results/split_manifest.json` (hashes of every split) and `results/data_summary.json` (counts and class balance). Why this revision and not the newer `combined_reduced.json`: see PROGRESS.md, Phase 2.

## Repository layout

```
frontend/       PhishGuard_AI.html, bert_connector.js
backend/        Flask API (app.py)
src/            data.py (data download + split); later evaluation and models
scripts/        check_env.py and later training/evaluation scripts
results/        committed evidence: split manifest, data summary, later metrics
docs/           project proposal
third_party/    Apache-2.0 licence of the reference project
PhishGuard_BERT/  earlier draft scaffold, kept until fully replaced
```

## Third-party work

This project builds on **Phishing Detection with BERT** by **Esteban Alvarado**, licensed under the Apache License 2.0:

- Code: https://github.com/EstAlvB/Phishing-Detection-with-BERT — cloned locally into `reference/` (not committed) at commit `0b0b1a8c86e141bec6a4a1b7afab9fe796cd4b73`
- Dataset: https://huggingface.co/datasets/ealvaradob/phishing-dataset — `train.json`/`test.json` at revision `23b8c17f4c9438c3e5d7729be349db049614a266` (MIT at that revision; Apache-2.0 later)
- Model: https://huggingface.co/ealvaradob/bert-finetuned-phishing (revision `fa8fb73a007174c410ab7160d4e4c6e6b8d998d4`)

To recreate the local reference copy:

```bash
git clone https://github.com/EstAlvB/Phishing-Detection-with-BERT reference
git -C reference checkout 0b0b1a8c86e141bec6a4a1b7afab9fe796cd4b73
```
