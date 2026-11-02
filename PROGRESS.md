# PhishGuard BERT — Progress Log

**Repo:** https://github.com/klokiri/phishguard-bert (public, chosen by Kasim on 2026-11-02)
**Build window:** Mon 2 Nov 2026 → Sun 8 Nov 2026 (7 days)

| Phase | Status | Tag |
|---|---|---|
| 0 Recon and repo setup | done | `phase-0-recon` |
| 1 Restructure and environment | done | `phase-1-env` |
| 2 Data | done | `phase-2-data` |
| 3 Evaluation and baselines | not started | |
| 4 BERT | not started | |
| 5 Backend | not started | |
| 6 Frontend | not started | |
| 7 Evaluation assets, report, cleanup | not started | |

---

## Phase 0 — Recon findings (2026-11-02)

### 1. Machine (this differs from the prompt's assumption)

The prompt assumed Windows with an NVIDIA GPU. The real machine:

| Item | Value | How checked |
|---|---|---|
| OS | macOS 26.5.1 (build 25F80), arm64 | `sw_vers`, `uname -m` |
| CPU / GPU | Apple M4 Pro, 20-core integrated GPU, Metal 4 | `sysctl`, `system_profiler SPDisplaysDataType` |
| Memory | 24 GB unified (CPU and GPU share it; there is no separate VRAM) | `sysctl hw.memsize` |
| NVIDIA / CUDA | **none** (`nvidia-smi: command not found`) | `nvidia-smi` |
| Python | system `/usr/bin/python3` = 3.9.6 (too old for current `transformers`); `uv` is installed at `~/.local/bin/uv`; Homebrew present | `python3 --version`, `which` |
| Free disk | 267 GiB | `df -h ~` |
| git / gh | git 2.39.5; gh 2.87.3, logged in as `klokiri` over HTTPS | `git --version`, `gh auth status` |
| Commit identity | `Kasim <97587117+klokiri@users.noreply.github.com>` (repo-local config) | set in Phase 0 with Kasim's choice |

**Consequences for the plan**
- PyTorch will run on Apple's **MPS** backend (Metal), not CUDA. `torch.cuda.is_available()` will be `False`; `check_env.py` must report `torch.backends.mps.is_available()` instead.
- `setup.bat` (Windows + CUDA 11.8 wheels) cannot be used at all. Phase 1 replaces it with a `uv`-managed Python 3.12 venv and the standard macOS arm64 PyTorch wheel.
- `fp16=True` in the HF `Trainer` has historically required CUDA. Mixed precision on MPS will be **tested in Phase 1**, not assumed; fp32 is the fallback.
- The "VRAM < 8 GB → DistilBERT" rule does not apply directly (24 GB unified memory). The real constraint is **speed**: MPS is slower than a desktop NVIDIA card. Phase 4 benchmarks a few hundred steps first and then gives a time estimate before any long run.

### 2. The bundle

- The folder held `PHISHGUARD_BERT_PROMPT.md`, `PhishGuard_AI.html` and `phishguard-ai-starter.zip`. `PhishGuard_BERT/` and `docs/` existed only inside the zip.
- The loose HTML and prompt are byte-identical to the zip copies (SHA-256 `ffefb5a9…` and `084f3a79…`).
- With Kasim's approval, `PhishGuard_BERT/` and `docs/` were unpacked unchanged (`diff -r` clean) and committed as the baseline. The zip is git-ignored, not deleted.
- **Secret scan:** clean. The only `sk-ant-` hits are the documented placeholders/validation (`PhishGuard_AI.html` lines 865, 1299, 1484, 1503). No `ghp_`/`gho_`/`hf_`/AWS keys, private keys or `.env` files.

**`PhishGuard_AI.html`** (2256 lines, vanilla JS + Chart.js 4.4.0 from cdnjs + localStorage): button `#analyse-btn` calls `runAnalysis()` (line 1063 → defined at 1645), which dispatches to `callClaudeAPI()` (1704) or `simulateAnalysis()` (1765). `escHtml()` exists at line 2242.

### 3. Scaffold defects — verified

| File | Defect (verified) |
|---|---|
| `data/prepare_dataset.py:26` | Loads config `"emails"`, which **does not exist** (the loading script defines only `texts`, `urls`, `webs`, `combined_full`, `combined_reduced`), and omits `trust_remote_code=True`. Worse than reported: on failure, lines 40+ **silently fall back to 16 hand-written emails** and carry on. |
| `data/prepare_dataset.py:86, 102-103` | Truncates every text to 1,000 characters with no justification; random 70/15/15 split ignores leakage. |
| `notebooks/day2_quick_test.py:31,33` | Says the model download is ~440 MB; `ealvaradob/bert-finetuned-phishing` is bert-large, `pytorch_model.bin` = 1.34 GB. (`day3_finetune.py:123` also says 440 MB, but that one is about `bert-base-uncased`, where it is roughly correct. `README.md` does not state a size.) |
| `notebooks/day6_evaluation.py:135` | `f1_score(..., weighted=True)` → `TypeError`; parameter is `average=`. |
| `backend/app.py:150-154` vs `day7_demo_and_report.py:46` | Thresholds disagree: app uses 0.5 cut then overrides 0.40–0.60 to "suspicious"; day7 uses >0.6 / >0.4. |
| `backend/app.py:192,195,198` | `np.random.randint` noise added to indicator scores, and scores invented for signals that did not fire. |
| `notebooks/day7_demo_and_report.py:110-114` | Hard-coded fallback metrics (0.943, 0.941, 0.938, 0.944, 0.987). |
| `day7…:150` and `README.md:60,179` | Unmeasured claims: heuristic "~74%", "~89%", "~85-90%", "BERT 94%+". |
| `day7…:214` | Claims "attention-weight-derived indicators" — not implemented. |
| `day7…:198` | Maps NFR-09 (test coverage) to model accuracy. |
| `README.md:52,175` | "18,000+ labelled emails" — the combined dataset is mixed URLs/emails/SMS/HTML, not just emails. |
| `setup.bat` | Windows-only, CUDA 11.8 wheels, unchecked version pins. Not usable on this Mac. |
| `frontend-patch/bert_connector.js:76,81` | Uses `innerHTML` for status bar; integration instructions override the button handler after the fact. Hard-coded `http://localhost:5000`. `app.py:33` allows CORS `*`, so `file://` (origin `null`) will probably work — **to be tested in Phase 6**. |

### 4. The proposal (docs/PhishGuardAI_Project_Proposal.docx)

- **Objective:** ≥ 3 model architectures; accuracy ≥ 95% and F1 ≥ 0.93 **on a held-out test set of at least 10,000 labelled samples**. (The author's combined test split has 15,417 rows, which satisfies the size condition if reproduced.)
- **USP:** explainable, indicator-level outputs; swappable ("modular") ML.
- **FR-01** submission + analysis (proposal says Random Forest default + SHAP top-10); **FR-02** user authentication (JWT); **FR-03** dashboard; **FR-04** admin model-training module (CSV upload, SSE progress, deploy button).
- **NFR-01** bcrypt ≥12 · **02** TLS 1.2+/HSTS · **03** parameterised queries · **04** p95 ≤ 3 s at 50 users · **05** dashboard p95 ≤ 1.5 s · **06** SUS ≥ 75 · **07** WCAG 2.1 AA · **08** 99.5% uptime · **09** ≥ 80% backend statement coverage · **10** immutable audit log · **11** rate limit 100/h per user, 20/h per IP, 429 + `Retry-After` · **12** `ModelService` interface + OpenAPI 3.0 spec.

**Gap to discuss with the supervisor (not hidden):** the 7-day plan covers FR-01, FR-03 (existing UI), NFR-03, NFR-09, NFR-11 (partial: per-IP limit), NFR-12 (partial: `ModelService`; OpenAPI spec only if time). It does **not** cover FR-02 (auth/JWT), FR-04 (web training module), NFR-01/02/05/06/07/08/10, PostgreSQL, or SHAP. The proposal named Random Forest as the default model; this build makes BERT primary with RF/TF-IDF models as baselines. These deviations must be stated in the report.

### 5. Reference project (Esteban Alvarado, Apache-2.0)

- Cloned `https://github.com/EstAlvB/Phishing-Detection-with-BERT` into `reference/` (git-ignored) at commit **`0b0b1a8c86e141bec6a4a1b7afab9fe796cd4b73`** (2024-01-31, "Update bert finetuning notebook").
- HF dataset `ealvaradob/phishing-dataset` at revision **`55caf9af01ae498d68ff44886ff69a4b165eb54b`**, licence apache-2.0. `combined_reduced.json` = 521,149,713 bytes, LFS SHA-256 `222fd5f841a0565841f8ede2e21b9b48cfea933a6b5c3a7da3e3a2cbc156d3d5`. The loading script just reads this JSON in file order into one `train` split, so **pandas can load it directly** — no `datasets<4` pin and no `trust_remote_code` needed. Phase 2 downloads only this file (well under the 2 GB ask-first limit).
- HF model `ealvaradob/bert-finetuned-phishing` at revision `fa8fb73a007174c410ab7160d4e4c6e6b8d998d4`, licence apache-2.0, base `bert-large-uncased`, `pytorch_model.bin` 1.34 GB.

#### 5a. How the author split the data (`finetuning-bert.ipynb`, cells 7, 27, 31)
```python
dataset = load_dataset("ealvaradob/phishing-dataset", "combined_reduced", trust_remote_code=True)
df = tokenized_dataset['train'].to_pandas()
train, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
```
- 80/20, **stratified on label, `random_state=42`**, applied to the dataset in its published row order. **`combined_reduced` is confirmed** as the config used.
- **No validation set.** The `Trainer` used `eval_dataset=test` with `load_best_model_at_end=True`, so the **test split was also used to pick the checkpoint**. Even on an exactly reproduced split, the author's figures are slightly optimistic. Kasim's models will carve validation from train only.
- Tokenisation: `bert-large-uncased` tokenizer, `truncation=True` (max 512 tokens), dynamic padding. lr 2e-5, batch 16, 4 epochs, weight decay 0.01, fp16, `torch_compile=True`.

#### 5b. Red flag: the split may not be exactly reproducible
Three sources imply three different dataset sizes:

| Source | Evidence | Implied train rows |
|---|---|---|
| HF model card | 3,866 steps/epoch × batch 16 | 61,841–61,856 |
| `phishing-classifiers.ipynb` | combined test set = 15,417 rows | 61,668 (total ≈ 77,085) |
| `finetuning-bert.ipynb` saved output | 6,680 steps at "epoch 2.0" although `num_epochs=4` | ≈ 53,440 |
| `preprocessing.ipynb` | `combined_reduced` shown with 77,677 rows (before upload) | 62,141 |

So the committed notebook output is not the run that produced the published model, and the model's training set seems to differ slightly (~180 rows) from what the published dataset gives. Also, the classifier notebook measured the published model at accuracy **0.9844** on its combined test split, while the model card reports **0.9717**. The higher number is *consistent with* some test rows having been seen in training (a hypothesis, not proven).

**Plan for Phase 2:** load the published JSON, count rows, apply the exact split, and compare with 15,417 / 61,668. If the counts match the classifier notebook but not the model card, the split cannot be proven identical to the model's training split → per the leakage rules, the pre-trained model's figures will be reported as **author-reported only**, and any number we measure for it on this split will be labelled "not independent". Kasim will be asked to choose before Phase 4a.

Also: the per-domain results in the classifier notebook (`texts`, `urls`, `webs` splits, e.g. "Texto 0.9958") come from **separate** splits that overlap the model's combined training data. They must not be cited.

#### 5c. Preprocessing (`preprocessing.ipynb`)
- **URLs** (`URLS.csv`): label inverted (source 0 = phishing → 1), renamed, shuffled **without a seed**, deduplicated → 808,042 rows (52.8% benign).
- **Emails** (Kaggle "phishingemails", Enron-based): "Phishing Email" → 1, "Safe Email" → 0; dedupe, drop NaN, drop literal `"empty"` → 17,536 rows (62.6% benign).
- **SMS** (Mendeley f45bkkt8pr): `ham` → 0, **`smishing` and `spam` → 1** (spam counted as phishing — a label-definition limitation to note); dedupe → 5,949; benign undersampled to ⌊4/3 × minority⌋ with seed 42 → 1,486 benign / 1,115 phishing.
- **Websites:** first 30,000 rows of an SQLite index; HTML files > 100 KB dropped; dedupe → 15,756 (65.1% benign); the websites' URLs added to the URL set; HTML minified with `minify_html`.
- **combined_reduced** = 5% stratified sample of URLs (seed 42) + emails + SMS + websites, shuffled with seed 42. Because the URL shuffle had no seed, the preprocessing cannot be re-run identically — we rely on the **published JSON** as the fixed artefact.

#### 5d. Classical classifiers (`phishing-classifiers.ipynb`)
- Same split recipe (`test_size=0.2, stratify, random_state=42`) for each dataset.
- Features: **TF-IDF only** — `TfidfVectorizer(ngram_range=(1,2), max_df=0.25, stop_words='english', max_features=50000)`, fitted on train only. `features.py` is **not** used by any classifier (only in `analysis.ipynb` for descriptive class averages).
- Models: XGBoost (`colsample_bytree=0.7, gamma=0.2, lr=0.1, max_depth=12, min_child_weight=2, n_estimators=100, subsample=0.8`), Multinomial NB (`alpha=0.01`), LSTM-CNN (Keras, TextVectorization 100 tokens, 4–12 epochs), BERT-large without fine-tuning, BERT fine-tuned.
- Metrics: accuracy, precision, recall, ROC-AUC (no F1, no FPR).
- **Author-reported** results on the combined test split (not measured by us):

  | Model | Accuracy | Precision | Recall | ROC-AUC |
  |---|---|---|---|---|
  | XGBoost | 0.8500 | 0.9199 | 0.7056 | 0.9422 |
  | MNB | 0.8750 | 0.8920 | 0.8004 | 0.9578 |
  | LSTM-CNN | 0.7250 | 0.9359 | 0.3730 | 0.8252 |
  | BERT-large, not fine-tuned | 0.5794 | 0.7188 | 0.0035 | 0.5879 |
  | BERT fine-tuned (published model) | 0.9844 | 0.9842 | 0.9788 | 0.9969 |

  Note for expectations: the author's own TF-IDF baselines are well below the proposal's ≥95% target. Our baselines may be too. Results will be reported as measured.

#### 5e. `features.py`
`URLFeatures` (19), `TextFeatures` (6), `HTMLFeatures` (23). Imports `pyquery` at module level → must be installed even for URL/text use.

### 6. Decisions recorded / proposed
- **F1 definition (proposed, confirm in Phase 3):** binary F1 for the phishing class (label 1), matching the author's binary precision/recall. Weighted F1 can be shown as a secondary column, clearly labelled.
- Test set = author's 20% split, frozen once in Phase 2 and used for every model.
- Thresholds will live in one config file (Phase 5).

---

## Phase 1 — Restructure and environment (2026-11-02)

**Restructure** (commit `9d08825`, `git mv` only, 0 content changes):
`PhishGuard_AI.html` → `frontend/`; `PhishGuard_BERT/frontend-patch/bert_connector.js` → `frontend/`; `PhishGuard_BERT/backend/app.py` → `backend/`. The rest of the scaffold stays in `PhishGuard_BERT/` until Phase 7.

**Environment**
- Python 3.12.12 installed with `uv python install 3.12`; venv at `.venv` (`uv venv --python 3.12 .venv`).
- `requirements.txt` pins the direct dependencies. Verified twice: in `.venv`, and in a throw-away fresh venv built only from `requirements.txt`; both pass `scripts/check_env.py` (exit code 0).
- Key versions: torch 2.15.0, transformers 5.22.1, accelerate 1.15.0, scikit-learn 1.9.1, xgboost 3.4.1, pandas 3.1.0.
- **System dependency:** xgboost failed to import until `brew install libomp` (OpenMP runtime). Documented in `requirements.txt` and README.
- `transformers` 5.x / `datasets` not installed: Phase 2 reads `combined_reduced.json` with pandas, so the loading-script problem in `datasets` 4.x does not arise.

**GPU check** (`.venv/bin/python scripts/check_env.py`):
- `torch.cuda.is_available()` = False (expected, no NVIDIA); `torch.backends.mps.is_available()` = True.
- 256×256 matmul on MPS matches the CPU result.
- autocast float16 and bfloat16 both work on MPS (forward + backward of a linear layer).
- Extra test (scratch script, not committed): HF `Trainer` ran 4 steps on a tiny random BERT on `mps` with `fp32`, `fp16=True` and `bf16=True` — all three ran. Whether fp16 is numerically stable at bert-base scale is **checked in Phase 4**, not assumed.

**Note on the definition of done:** it says "`check_env.py` shows the GPU detected". On this machine that means MPS = True, not CUDA = True.

---

## Phase 2 — Data (2026-11-02)

### The split question from Phase 0 §5b is resolved

1. Applying the author's recipe (`train_test_split(test_size=0.2, stratify, random_state=42)`) to today's `combined_reduced.json` gives 77,677 rows → 62,141 train (3,884 steps/epoch) / 15,536 test. This matches **neither** the model card (3,866 steps/epoch) nor the classifier notebook (15,417 test rows).
2. The Hub commit history explains it. The model's final training runs are dated 2023-12-27 22:16–23:08 UTC (model repo commits `ecbff15e`…`a14076d9`, then "load 4 epochs model (best results)"). At that time the dataset (revision `23b8c17f`, 2023-12-27 19:11 UTC) contained a pre-made `train.json` + `test.json`. `combined_reduced.json` was uploaded later, on 2024-01-31.
3. **Verified:** `train.json` at `23b8c17f` has **61,849 rows → 3,866 steps/epoch at batch 16 — an exact match with the model card.** So this `train.json` is the published model's training set and the matching `test.json` (15,466 rows) is its held-out set.
4. Today's file keeps nearly the same emails/SMS (19,114 of 20,184 match after normalising case/whitespace) and HTML (12,624 of 15,745), but only 2,140 of its 41,748 URLs appear in the old files: the 5% URL sample was re-drawn (the author's URL shuffle had no seed).
5. Consequence: on a split of today's file, **5,986 of 15,536 test rows (38.5%) are in the published model's training data** (normalised match). Evaluating the model there would be heavily inflated.

**Decision (Kasim, 2026-11-02):** use the original split at revision `23b8c17f4c9438c3e5d7729be349db049614a266` for **every** model. test = `test.json` unchanged; train/val = `train.json` split 90/10, stratified, seed 42.

What this means for the report:
- The pre-trained model gets a **genuinely held-out** test set (its own). Caveat: the author used this test set during training for per-epoch evaluation and to pick the best checkpoint (`eval_dataset=test`, `load_best_model_at_end=True`), so its score on it is still slightly optimistic. Our own models never see the test set before final evaluation.
- The author's classifier-notebook figures (Phase 0 §5d) were computed on a different dataset version (15,417 test rows), so they are **not** directly comparable with our results; cite them only as author-reported context.
- The dataset licence at this revision is **MIT** (it became Apache-2.0 later); both permit this use. NOTICE updated.

### Result (`.venv/bin/python -m src.data`)

| Split | Rows | Phishing | Benign | Phishing fraction |
|---|---|---|---|---|
| train | 55,664 | 23,431 | 32,233 | 0.4209 |
| val | 6,185 | 2,603 | 3,582 | 0.4209 |
| test | 15,466 | 6,510 | 8,956 | 0.4209 |

(read from `results/data_summary.json`)

- Both downloaded files pass their SHA-256 checks; `split_manifest.json` records file and content hashes. Running the script twice gives an identical manifest (verified with `diff`).
- Sample types (heuristic `guess_type`, test set): 8,273 URL, 4,041 text (email/SMS), 3,152 HTML.
- Text length is extremely skewed: median 80 characters, 90th percentile 25,256, maximum 17,036,692 (one huge HTML page). Phase 4 must choose `max_length` from token statistics, and Phase 3's feature extraction must cope with very long inputs.
- **Limitation (kept, reported):** 22 test rows have an exact duplicate text in `train.json` (68 after normalising case and whitespace = 0.44% of test). They are kept so the test set stays identical to the author's.
- `data/raw/combined_reduced.json` (497 MB, git-ignored) was downloaded during the investigation and is no longer used. It can be deleted.
- The test set (15,466 rows) satisfies the proposal's "held-out test set of at least 10,000 labelled samples".

---

## 7-day schedule (realistic for an M4 Pro on MPS)

| Day | Date | Work | Done when |
|---|---|---|---|
| 1 | Mon 2 Nov | Phase 0 (this) · Phase 1: `git mv` restructure, uv venv (Py 3.12), PyTorch for MPS, `requirements.txt`, `check_env.py`, quick MPS precision test | `check_env.py` shows MPS available; tags `phase-0`, `phase-1` |
| 2 | Tue 3 Nov | Phase 2: download `combined_reduced.json` (521 MB), reproduce split, validation from train, manifest + summary. **Decision point:** split reproducible? Phase 3 start: `evaluate.py` + ported JS heuristic | manifest committed; heuristic results JSON |
| 3 | Wed 4 Nov | Phase 3: TF-IDF + LogReg, TF-IDF + XGBoost/MNB (author's setup), RF on `features.py` features · Phase 4a: pre-trained BERT-large inference on test (~15k rows; benchmark first) | ≥3 model results JSONs on one test set |
| 4 | Thu 5 Nov | Phase 4b: token-length stats → `max_length`; benchmark 200 steps; **ask before full run**; fine-tune `bert-base-uncased` (run overnight if > 15 min) | model saved locally; history JSON committed |
| 5 | Fri 6 Nov | Phase 5: config, `ModelService`, deterministic indicators, occlusion attribution, pytest + coverage | tests pass; `results/coverage.json` |
| 6 | Sat 7 Nov | Phase 6: connect HTML to backend; test via `http.server` and `file://` | BERT primary, heuristic fallback working |
| 7 | Sun 8 Nov | Phase 7: figures, error analysis, report draft, README, remove scaffold. Buffer for overruns | definition of done met |

**Risks:** (1) MPS training speed — mitigation: benchmark first, shorter `max_length`, or DistilBERT; (2) split not reproducible — mitigation: author-reported labelling (§5b); (3) proposal scope gap (§4) — mitigation: state it plainly in the report.
