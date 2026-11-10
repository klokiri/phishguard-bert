# PhishGuard AI — BERT edition

Final-year capstone by Kasim: an explainable phishing detector for emails and URLs, with a BERT model, classical baselines, a Flask backend and the PhishGuard AI web front end.

**Results:** [results/figures/comparison_table.md](results/figures/comparison_table.md) compares all six models on one held-out test set, and the figures are in [results/figures/](results/figures/). [results/report_draft.md](results/report_draft.md) is a draft of the methods, results, error analysis and limitations, generated from the results files. This README deliberately contains no metric values: every number lives in, and is read from, `results/`.

**Project log:** [PROGRESS.md](PROGRESS.md) holds the decisions and findings for every phase. [AI_ASSISTANCE_LOG.md](AI_ASSISTANCE_LOG.md) declares which files were AI-generated.

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

## Running the backend

```bash
.venv/bin/python -m backend.app
```

Serves the fine-tuned BERT model at http://127.0.0.1:5000/api/v1 (endpoints: `health`, `model/info`, `analyse`, `batch`, `history`, `stats`). To swap models without changing code, set `PHISHGUARD_MODEL=heuristic` or `PHISHGUARD_MODEL=classical`. All settings are in `src/config.py`. Each response contains the phishing probability, rule-based indicators with their evidence and data-derived scores, the words that most changed the model's score (leave-one-word-out occlusion), and an explanation built only from what fired.

## Running the web app

Start the backend (above), then either open `frontend/PhishGuard_AI.html` directly in a browser, or serve it:

```bash
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory frontend
```

and open http://localhost:8000/PhishGuard_AI.html. Log in with the pre-filled demo account. The Analyse page shows whether the BERT backend is online. If it isn't, the app falls back to the in-browser heuristic, or to the Claude API if you enter a key. The "Registered Models" list shows real measured metrics from `frontend/model_results.js`, which is generated from `results/*.json`:

```bash
.venv/bin/python scripts/export_frontend_results.py
```

## Reproducing the results

Every number comes from a file in `results/`, produced by one command:

| Results file | Command |
|---|---|
| `results/split_manifest.json`, `results/data_summary.json` | `.venv/bin/python -m src.data` |
| `results/heuristic_parity.json` (needs Node.js) | `.venv/bin/python scripts/check_heuristic_parity.py` |
| `results/heuristic.json` | `.venv/bin/python scripts/run_heuristic.py` |
| `results/tfidf_logreg.json` | `.venv/bin/python scripts/train_classical.py tfidf_logreg` |
| `results/tfidf_xgboost.json` | `.venv/bin/python scripts/train_classical.py tfidf_xgboost` |
| `results/features_rf.json` | `.venv/bin/python scripts/train_classical.py features_rf` |
| `results/bert_pretrained.json` (downloads the 1.2 GB author model) | `.venv/bin/python scripts/eval_pretrained_bert.py` |
| `results/token_length_stats.json` | `.venv/bin/python scripts/token_length_stats.py` |
| `results/bert_base.json`, `results/bert_base_history.json` (~1.5 h on an M4 Pro; Mac plugged in, lid open) | `caffeinate -i -s .venv/bin/python scripts/train_bert.py --max-length 512 --epochs 3 --precision fp16` |
| `results/bert_base_training_config.json` (after training) | `.venv/bin/python scripts/export_training_config.py` |
| `results/indicator_stats.json` (training split only) | `.venv/bin/python scripts/build_indicator_stats.py` |
| `results/coverage.json` (NFR-09) | `.venv/bin/python scripts/coverage_report.py` |
| `results/scheme_artefact.json` (one model per run) | `.venv/bin/python scripts/scheme_artefact.py <model>` for each of `heuristic tfidf_logreg tfidf_xgboost features_rf bert_base bert_pretrained` |
| `results/error_analysis.json` | `.venv/bin/python scripts/error_analysis.py` |
| `results/error_analysis_notes.json` | Hand-written (by the AI assistant, for Kasim's review): one note per sampled error |
| `results/ood_manifest.json` (downloads ~26 MB of public email corpora; email text stays in git-ignored `data/ood/`) | `.venv/bin/python -I scripts/build_ood.py` |
| `results/ood_<model>.json` (one model per run) | `.venv/bin/python scripts/eval_ood.py <model>` for each of `heuristic tfidf_logreg tfidf_xgboost features_rf bert_base bert_pretrained` |
| `results/figures/*` (table + 5 figures) | `.venv/bin/python scripts/make_figures.py` |
| `results/report_draft.md` | `.venv/bin/python scripts/make_report.py` |
| `frontend/model_results.js` | `.venv/bin/python scripts/export_frontend_results.py` |

Browser test of the page opened from disk (`file://`): see the header of `scripts/browser_test.mjs`.

Model weights are not in git. `models/phishguard-bert-base/` is recreated by the training command and `models/pretrained/` by the evaluation script. GPU training on Apple MPS is not bit-for-bit deterministic, so a re-run gives very close but not necessarily identical numbers.

Each model also writes per-row test scores to `results/predictions/<model>.csv.gz`. Tests: `.venv/bin/python -m pytest tests/`.

Metric definitions: precision, recall and F1 are for the phishing class (binary); FPR = FP / (FP + TN); see `src/evaluate.py`.

## Repository layout

```
frontend/       PhishGuard_AI.html, bert_connector.js, model_results.js (generated)
backend/        Flask API: app.py, model_service.py, indicators.py, attribution.py, explain.py
src/            data.py (split), evaluate.py (shared metrics), features.py (from reference), models/
tests/          pytest tests
scripts/        environment check, training, evaluation, figures, report, browser test
results/        committed evidence: split manifest, metrics, predictions, figures, report draft
docs/           project proposal
third_party/    Apache-2.0 licence of the reference project
```

## Build timeline (git tags)

Each phase ends with an annotated tag. Check one out with `git checkout <tag>`.

| Tag | Date | Commit | What was completed |
|---|---|---|---|
| `phase-0-recon` | 2026-11-02 | `ebfcb68` | Recon, baseline import of the starting files, GitHub repo, schedule |
| `phase-1-env` | 2026-11-02 | `b7fdd69` | Repository layout, Python 3.12 environment, pinned requirements, GPU (MPS) check |
| `phase-2-data` | 2026-11-02 | `8bb82cc` | The published model's original train/test split, validation set, split manifest |
| `phase-3-baselines` | 2026-11-07 | `6115ee7` | Shared evaluation, heuristic baseline (JS parity), three classical models |
| `phase-4-bert` | 2026-11-09 | `6991504` | Published BERT evaluated on its held-out set; bert-base fine-tuned |
| `phase-5-backend` | 2026-11-09 | `2b19147` | Config, ModelService, data-derived indicators, occlusion attribution, tests, coverage |
| `phase-6-frontend` | 2026-11-10 | `f4e46aa` | Web app connected to the BERT backend with fallback; invented metrics removed |
| `phase-7-report` | 2026-11-10 | `d16ba66` | Figures, scheme-artefact test, error analysis, report draft, scaffold removed |
| `ood-eval` | 2026-11-11 | (latest) | Out-of-distribution email test set (Nazario phishing 2023–24, SpamAssassin ham) for all six models |

## Third-party work

This project builds on **Phishing Detection with BERT** by **Esteban Alvarado**, licensed under the Apache License 2.0:

- Code: https://github.com/EstAlvB/Phishing-Detection-with-BERT — cloned locally into `reference/` (not committed) at commit `0b0b1a8c86e141bec6a4a1b7afab9fe796cd4b73`
- Dataset: https://huggingface.co/datasets/ealvaradob/phishing-dataset — `train.json`/`test.json` at revision `23b8c17f4c9438c3e5d7729be349db049614a266` (MIT at that revision; Apache-2.0 later)
- Model: https://huggingface.co/ealvaradob/bert-finetuned-phishing (revision `fa8fb73a007174c410ab7160d4e4c6e6b8d998d4`). It's evaluated here as a third-party baseline, and it is not this project's model.
- `src/features.py` is his `features.py`, copied unchanged with an attribution header.

Out-of-distribution test emails: Jose Nazario's phishing corpus (CC BY 4.0) and the SpamAssassin public corpus. These are downloaded by `scripts/build_ood.py` and not redistributed here; see [NOTICE](NOTICE).

Full licence text: [third_party/LICENSE-EstAlvB.md](third_party/LICENSE-EstAlvB.md); notice: [NOTICE](NOTICE). The "Use of third-party work" paragraph for the report is section 7 of `results/report_draft.md`.

To recreate the local reference copy:

```bash
git clone https://github.com/EstAlvB/Phishing-Detection-with-BERT reference
git -C reference checkout 0b0b1a8c86e141bec6a4a1b7afab9fe796cd4b73
```
