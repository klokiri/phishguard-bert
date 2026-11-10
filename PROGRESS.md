# PhishGuard BERT — Progress Log

**Repo:** https://github.com/klokiri/phishguard-bert (public, chosen by Kasim on 2026-11-02)
**Build window:** Mon 2 Nov 2026 → Sun 8 Nov 2026 (7 days)

| Phase | Status | Tag |
|---|---|---|
| 0 Recon and repo setup | done | `phase-0-recon` |
| 1 Restructure and environment | done | `phase-1-env` |
| 2 Data | done | `phase-2-data` |
| 3 Evaluation and baselines | done | `phase-3-baselines` |
| 4 BERT | done | `phase-4-bert` |
| 5 Backend | done | `phase-5-backend` |
| 6 Frontend | done | `phase-6-frontend` |
| 7 Evaluation assets, report, cleanup | done | `phase-7-report` |

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

## Phase 3 — Shared evaluation and baselines (2026-11-07)

**Schedule note:** work resumed on 7 Nov (day 6 of 7), so Phases 3–7 are compressed into the remaining time. Cut order from the prompt applies if needed (OOD set first, then token attribution).

**`src/evaluate.py`** — one function for every model, on the one test set (15,466 rows). Metric decision, used everywhere from now on: **precision/recall/F1 are binary for the phishing class (label 1)**; weighted F1 is stored as `f1_weighted`, labelled. Also FPR = FP/(FP+TN), ROC-AUC from the continuous score, confusion matrix, mean latency per sample, and all metrics per sample type. Each result file records the git commit it was produced from. A suffix `+dirty` means it was generated from code that was not yet committed; that code is in the same commit that adds the results file. Tests: `tests/test_evaluate.py` (hand-computed values).

**Baseline 0 — the app's heuristic** (`src/models/heuristic.py`): port of `simulateAnalysis()`. **Parity proven:** the original JS, extracted from the HTML and run in Node.js, gives the same score and verdict as the port on **all 15,466 test rows (0 mismatches, `results/heuristic_parity.json`)**. Non-fired indicators score 0 instead of `Math.random()`.

**Classical models** (`src/models/classical.py`, `scripts/train_classical.py`), fixed hyperparameters (no tuning), TF-IDF fitted on train only:
- `tfidf_logreg`: author's TF-IDF settings + Logistic Regression.
- `tfidf_xgboost`: author's TF-IDF settings + XGBoost with the author's hyperparameters.
- `features_rf`: Random Forest (300 trees) on the reference `features.py` features (copied to `src/features.py` unchanged, with attribution header). 0 HTML parse failures.

### Results on the shared test set (generated from `results/*.json`)

| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | Latency (ms/sample) | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| `heuristic` | 0.5792 | 1.0 | 0.0003 | 0.0006 | 0.0 | 0.5897 | 0.3206 | – |
| `tfidf_logreg` | 0.8842 | 0.906 | 0.8088 | 0.8546 | 0.061 | 0.9627 | 0.3633 | 38.4 |
| `tfidf_xgboost` | 0.846 | 0.9131 | 0.7008 | 0.793 | 0.0485 | 0.9393 | 0.4154 | 245.4 |
| `features_rf` | 0.8735 | 0.873 | 0.8184 | 0.8448 | 0.0865 | 0.9328 | 4.0027 | 189.2 |

| Model | URL acc | Text acc | HTML acc |
|---|---|---|---|
| `heuristic` | 0.5332 | 0.6169 | 0.6516 |
| `tfidf_logreg` | 0.8448 | 0.9423 | 0.9131 |
| `tfidf_xgboost` | 0.7812 | 0.9124 | 0.9308 |
| `features_rf` | 0.8837 | 0.8349 | 0.8959 |

**Observations**
- The heuristic gives only 2 "phishing" verdicts in 15,466 rows. Its accuracy (0.579) equals the benign share of the test set, so it is no better than always answering "legitimate". The hand-checked test shows why: `http://paypa1-login.tk/x` fires four risk rules and still scores 0.549, under the 0.55 cut-off.
- None of the classical models reaches the proposal's target (accuracy ≥ 0.95, F1 ≥ 0.93). Best: TF-IDF + LogReg, accuracy 0.8842 / F1 0.8546.
- XGBoost (0.846) is close to the author's reported 0.850 for the same setup on a similar dataset version — a sanity check that the pipeline matches his.
- URLs are the weak spot for TF-IDF models (single "words"); the hand-crafted features handle URLs best (0.8837) but are weaker on email/SMS text.
- RF latency (4.0 ms/sample) is dominated by feature extraction, mostly HTML parsing.

**Note on `git_commit` stamps in results files** (found in Phase 4): until commit `a7804bd`, `src/evaluate.py` detected uncommitted changes with `git diff HEAD`, which ignores new (untracked) files. So the Phase 3 and 4a results are stamped with the commit *before* the one that added the code that produced them: `heuristic.json` → code in `a7d33da`; `tfidf_*.json`, `features_rf.json` → `6115ee7`; `bert_pretrained.json` → `a7804bd`. Fixed in Phase 4 (`git status --porcelain`), so later files are stamped `+dirty` when generated before their code is committed.

---

## Phase 4 — BERT (2026-11-07)

### 4a. The author's pre-trained model on the held-out test set

`.venv/bin/python scripts/eval_pretrained_bert.py` — `ealvaradob/bert-finetuned-phishing` @ `fa8fb73a` (bert-large, 1.2 GB), MPS, float16, batch 32, truncation at 512 tokens, length-sorted batches.

| | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|
| Measured by us (`results/bert_pretrained.json`) | 0.9716 | 0.9656 | 0.967 | 0.9663 | 0.025 | 0.9944 |
| Author-reported (model card, epoch 4) | 0.9717 | 0.9658 | 0.967 | not reported | 0.0249 | not reported |

- Our independent run reproduces the model card to within 0.0002 on every reported metric. This also confirms that the Phase 2 test set is the author's real held-out set.
- **How to describe it in the report:** measured on data the model was not trained on, but the author used this test set to pick the best epoch, so it is slightly optimistic. It is the author's model, not Kasim's.
- It is the only model so far above the proposal's target (accuracy ≥ 0.95, F1 ≥ 0.93).
- By sample type (accuracy / recall / FPR): URL 0.9758 / 0.971 / 0.020; text 0.9889 / 0.9819 / 0.0068; **HTML 0.9385 / 0.9317 / 0.0579** — HTML is the weakest, consistent with 91% of HTML pages being truncated at 512 tokens.
- float16 vs float32 on 400 rows: largest probability difference 0.00797, 0 changed decisions.
- Latency: 17.7 ms/sample batched; one request at a time (what the API sees) mean 74.9 ms, p95 184.6 ms, max 213.5 ms over 50 requests — well inside NFR-04's 3,000 ms (single user; not a 50-user load test).

### 4b. Fine-tuning Kasim's own `bert-base-uncased`

**Token lengths** (`results/token_length_stats.json`, train split, bert-base-uncased tokenizer): median 35 tokens overall, but very different per type. URLs: 97% fit in 64 tokens. Emails/SMS: median 201; 58% fit in 256, 79% in 512. HTML: median 7,318; only 8.6% fit in 512.

**Benchmark** (`scripts/train_bert.py --benchmark`, 60 steps, fp16, batch 16, length-grouped batches): 0.357 s/step at max_length 256 (~21 min/epoch), 0.442 s/step at 512 (~26 min/epoch); loss decreasing normally in fp16.

**Decision (Kasim, 2026-11-07):** max_length **512**, up to 3 epochs with early stopping on validation F1 (patience 1). Rationale: only ~25% slower than 256, reads 79% of emails in full instead of 58%, and matches the author model's input limit, which keeps the comparison fair.

transformers 5.x note: `group_by_length=True` is now `train_sampling_strategy="group_by_length"`.

**First run lost (7–9 Nov).** Started 7 Nov ~15:40. The macOS power log shows "Clamshell Sleep" (lid closed, on battery at 18%) at 16:33. Training froze at step 407 / 10,437 and the job was stopped on 9 Nov. Running speed between sleeps was normal. Two mistakes on our side: (1) the app's keep-awake only lasts while the session is active and cannot prevent lid-close sleep; (2) checkpoints were saved only once per epoch (every 3,479 steps), so nothing was kept. **Fix:** validation and checkpoints every 1,000 steps (early-stopping patience 3 evaluations ≈ 1 epoch), and the run wrapped in `caffeinate -i -s` on AC power. Restarted from scratch on 9 Nov.

**Second attempt lost (9 Nov, ~step 253).** The process grew to 22 GB on the 24 GB Mac, with 12 GB in swap, so steps took minutes. Cause: Apple's MPS backend caches GPU buffers per tensor shape, and free-length padding produced hundreds of shapes. The 60-step benchmark was too short to reveal it. **Fix:** pad batches to a multiple of 64 tokens (at most 8 shapes; padding is masked, so results are unaffected), empty the MPS cache every 50 steps, and log memory. A 400-step test held at 9.7 GB.

**Schedule:** the planned 7-day window ended on 8 Nov. Remaining phases (5 backend, 6 frontend, 7 report) continue after Phase 4.

### 4b result (third run, 9 Nov, completed)

Command: `caffeinate -i -s .venv/bin/python scripts/train_bert.py --max-length 512 --epochs 3 --precision fp16`. Training time 5,455 s (1 h 31 min), all 3 epochs, memory flat at 9.7 GB throughout. Full configuration: `results/bert_base_training_config.json` (exported from the saved `training_args.bin`: AdamW fused, lr 2e-5 linear decay, no warm-up, batch 16, weight decay 0.01, fp16, seed 42 / data_seed 42).

**Validation history** (from `results/bert_base_history.json`; validation set only, NOT test):

| Step | Epoch | Val loss | Val accuracy | Val precision | Val recall | Val F1 |
|---|---|---|---|---|---|---|
| 1000 | 0.287 | 0.2048 | 0.9287 | 0.8998 | 0.9347 | 0.9169 |
| 2000 | 0.575 | 0.2548 | 0.9167 | 0.9833 | 0.8160 | 0.8919 |
| 3000 | 0.862 | 0.1598 | 0.9486 | 0.9719 | 0.9040 | 0.9367 |
| 4000 | 1.150 | 0.1549 | 0.9559 | 0.9460 | 0.9493 | 0.9477 |
| 5000 | 1.437 | 0.1569 | 0.9593 | 0.9693 | 0.9328 | 0.9507 |
| 6000 | 1.725 | 0.2143 | 0.9484 | 0.9880 | 0.8882 | 0.9355 |
| 7000 | 2.012 | 0.1459 | 0.9635 | 0.9783 | 0.9339 | 0.9556 |
| 8000 | 2.300 | 0.1414 | 0.9688 | 0.9624 | 0.9635 | 0.9629 |
| 9000 | 2.587 | 0.1476 | 0.9688 | 0.9681 | 0.9574 | 0.9627 |
| 10000 | 2.874 | 0.1474 | 0.9690 | 0.9767 | 0.9489 | 0.9626 |
| **10437** | **3.000** | **0.1393** | **0.9715** | **0.9716** | **0.9604** | **0.9660** |

- **Selected checkpoint: `checkpoint-10437`** (end of epoch 3), the highest validation F1 (0.9660). Early stopping never triggered (the longest run without improvement was 2 evaluations).
- Because the best checkpoint is the last one, the model may not have fully converged; more epochs *might* help. This was **not** tested, and testing it by looking at test scores would be tuning on the test set. Report it as a limitation / future work.
- Training loss fell from 0.587 (step 100) to 0.028 (step 10,400) while validation loss stayed around 0.14: a clear train/validation gap, but validation F1 was still rising, so the gap did not hurt validation performance within 3 epochs.

**Test-set evaluation** (`results/bert_base.json`, run once after training, via `src/evaluate.py`):

| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | Latency (ms/sample, batched) |
|---|---|---|---|---|---|---|---|
| Heuristic (current app) | 0.5792 | 1.0 | 0.0003 | 0.0006 | 0.0 | 0.5897 | 0.3206 |
| TF-IDF + LogReg | 0.8842 | 0.906 | 0.8088 | 0.8546 | 0.061 | 0.9627 | 0.3633 |
| TF-IDF + XGBoost | 0.846 | 0.9131 | 0.7008 | 0.793 | 0.0485 | 0.9393 | 0.4154 |
| Random Forest (features) | 0.8735 | 0.873 | 0.8184 | 0.8448 | 0.0865 | 0.9328 | 4.0027 |
| **bert-base, fine-tuned by Kasim** | **0.9655** | **0.9631** | **0.9547** | **0.9589** | **0.0266** | **0.9935** | 6.2714 |
| bert-large, author's model † | 0.9716 | 0.9656 | 0.967 | 0.9663 | 0.025 | 0.9944 | 17.7278 |

† Author's model: not trained on this test set, but the author used it to choose his best epoch, so slightly optimistic.

Confusion matrix, bert-base (test, n = 15,466): TN 8,718 · FP 238 · FN 295 · TP 6,215.

F1 by sample type: bert-base URL 0.9689 · text 0.9828 · **HTML 0.8905** (author's model: 0.974 · 0.9854 · 0.9134).

**Interpretation**
- Kasim's bert-base **meets the proposal's objective** on the held-out test set: accuracy 0.9655 ≥ 0.95 and F1 0.9589 ≥ 0.93.
- It is within 0.006 accuracy and 0.0074 F1 of the author's bert-large while having ~3× fewer parameters, training on 10% less data, and running ~2.8× faster per sample (6.3 vs 17.7 ms batched).
- Test F1 (0.9589) is a little below best validation F1 (0.9660). That is expected: the checkpoint was chosen *because* it scored well on validation, so validation slightly overstates. The test number is the one to report.
- HTML is the weak spot for both BERT models (91% of pages are cut off at 512 tokens).

---

## Phase 5 — Backend (2026-11-09)

Run: `.venv/bin/python -m backend.app` (BERT; `PHISHGUARD_MODEL=heuristic` or `classical` to swap). Bound to 127.0.0.1:5000.

**Structure**
- `src/config.py` — the one config: model kind/paths (env vars), `max_length` 512, **decision threshold 0.5** (the rule every `results/*.json` was measured with; `src/evaluate.py` imports it), display-only "suspicious" band 0.4–0.6, rate limit 100/hour/IP, 100,000-character input limit, attribution word limit, DB path.
- `backend/model_service.py` — `ModelService` interface; `BertService`, `ClassicalService`, `HeuristicService`; `create_service()` (NFR-12).
- `backend/app.py` — `create_app(config, service)`. Validation → 422; non-JSON → 400; per-IP sliding-window limit → 429 + `Retry-After`; parameterised SQLite queries only; only a SHA-256 hash of the input is stored (no content); CORS limited to localhost and `file://` (`null`) origins.
- `backend/indicators.py` + `scripts/build_indicator_stats.py` — deterministic rule-based indicators; `backend/attribution.py` — occlusion; `backend/explain.py` — explanation text.

**Indicators — how scores are derived.** Each rule is run over the **training split**; for samples where it fires, the phishing rate is recorded (`results/indicator_stats.json`). An indicator appears only when its rule fires on the input; score = phishing rate × 100 if above the base rate (risk), else legitimate rate × 100 (safe). Base phishing rates: URL 0.4678, email/SMS 0.3838.

| Rule | Input | Training hits | Phishing rate | Shown? |
|---|---|---|---|---|
| Raw IP address as host | url | 306 | 0.9967 | yes, risk |
| '@' symbol in URL | url | 168 | 0.9762 | yes, risk |
| URL shortening service | url | 1,835 | 0.3804 | no (within 10 pts of base) |
| Hyphen in domain name | url | 3,948 | 0.7976 | yes, risk |
| Redirect-style '//' in path | url | 89 | 0.9775 | yes, risk |
| Explicit port number | url | 81 | 0.9012 | yes, risk |
| 'login' in URL | url | 1,260 | 0.9778 | yes, risk |
| 'admin' in URL | url | 279 | 0.9462 | yes, risk |
| Percent-encoded characters | url | 606 | 0.3383 | yes, safe |
| Long URL (over 75 characters) | url | 3,593 | 0.5026 | no (within 10 pts of base) |
| Many subdomains (3+ dots in host) | url | 3,485 | 0.798 | yes, risk |
| Many digits (more than 5) | url | 6,308 | 0.3844 | no (within 10 pts of base) |
| Frequently abused top-level domain | url | 2,872 | 0.9641 | yes, risk |
| Brand look-alike spelling | url | 7 | 1.0 | no (< 30 hits) |
| Unencrypted http:// | url | 3,047 | 0.7325 | no (excluded: dataset artefact) |
| Uses https:// | url | 4,864 | 0.8351 | no (excluded: dataset artefact) |
| Urgency language | email | 913 | 0.5717 | yes, risk |
| Request for credentials or payment details | email | 422 | 0.6137 | yes, risk |
| Threat or consequence language | email | 190 | 0.3895 | no (within 10 pts of base) |
| Prize or money lure | email | 1,189 | 0.6577 | yes, risk |
| Generic greeting | email | 48 | 0.9583 | yes, risk |
| Contains a link | email | 3,407 | 0.2486 | yes, safe |
| Link to a raw IP address | email | 72 | 0.9722 | yes, risk |
| Contains an email address | email | 1,854 | 0.2335 | yes, safe |
| Brand look-alike spelling | email | 0 | – | no (< 30 hits) |

**Important dataset finding (for the report's limitations).** Only **10.2%** of benign training URLs include a scheme (`http://`/`https://`) versus **45.1%** of phishing URLs (computed on `train.parquet`, URL rows). This is an artefact of how the source URL lists were collected, so "has https://" looks like a phishing sign in this data. Real users nearly always paste full URLs, so every model trained on this data may over-flag legitimate full URLs. The scheme rules are therefore hidden from users; the effect on the models is checked in the Phase 7 error analysis. Some other rates are counter-intuitive for the same kind of reason (e.g. "contains a link" is 75% legitimate because many benign emails are Enron business mail with links). Indicators show what correlates with phishing *in this dataset*.

**Token attribution.** Leave-one-word-out occlusion on the first 60 words, effect in log-odds (plain probability differences vanish when the model is ~0.9999 sure), model-agnostic. Shown as "occlusion (leave-one-word-out), effect in log-odds" — not attention, not SHAP.

**Explanation text** is built only from the probability, the rules that fired (with evidence) and the measured word effects. Extreme probabilities are shown as ">99.9%"/"<0.1%".

**Tests and coverage (NFR-09):** `.venv/bin/python scripts/coverage_report.py` → **62 passed; backend statement coverage 96.9% total, lowest module 95.7% (`app.py`); target 80% met** (`results/coverage.json`). Every API test runs against a fake model and the real heuristic with no code change (NFR-12 check). Bugs found by the tests: (1) the reference `URLFeatures.host_is_ip()` misses IP hosts with a port, so our rule checks the host name; (2) the full test run segfaulted because xgboost (Homebrew libomp) and torch (bundled libomp) were loaded in one process — xgboost is now imported lazily.

**Live check** (real server + curl): `/health` → `phishguard-bert-base` on `mps`; a phishing email with `Origin: null` (as sent by a `file://` page) → 200 with `Access-Control-Allow-Origin: null`, p = 0.9999, 4 indicators, 156 ms; empty content → 422; listening on 127.0.0.1 only.

**Latency (single user, not a load test):** typical request 25–215 ms including attribution; worst case seen, a long HTML page (60 words × 512 tokens re-scored), 973 ms. NFR-04's 50-user load test was not run.

**Not done (state in the report):** no authentication (FR-02, NFR-01), no TLS (NFR-02), no immutable audit log (NFR-10), no OpenAPI spec file (part of NFR-12), rate limit is per IP only (no per-user limit, as there are no users), no load test (NFR-04).

---

## Phase 6 — Frontend (2026-11-09 to 2026-11-10)

**Integration** (`frontend/PhishGuard_AI.html`, `frontend/bert_connector.js`)
- The connector only defines functions; nothing overrides the existing `onclick="runAnalysis()"`. `runAnalysis()` picks the engine: **local BERT backend first**; if it is offline or errors, the app's existing path (Claude API if the user set a key, else the in-browser heuristic). The fallback reason is shown on the result card.
- Backend status badge on the Analyse page (checked at start and every 30 s).
- New "Model Details" block: engine, model name, phishing probability (">99.9%"/"<0.1%" at the extremes), server latency, and the words that most changed the model's score with the method named ("occlusion (leave-one-word-out), effect in log-odds").
- Indicators show their evidence and data basis (tooltip); "safe" indicators have green bars; when nothing fires the card says so.
- All engine-derived text is inserted with `textContent` or `escHtml()`. `escHtml()` now escapes `'` and handles non-strings. Previously unescaped spots fixed: indicator names in `renderResult`, verdict and top indicator in the history/dashboard tables.

**Invented numbers removed from the app (Kasim's decision).** The original UI contained (1) a "Model Training" page that simulated a training run and displayed *random* metrics (accuracy 0.92–0.985 etc.), then saved them as a deployed model, and (2) a hard-coded model registry and "Active Model" card claiming Claude accuracy 0.992 / F1 0.991 / ROC 0.998. Both are gone: the training button now explains that web training (FR-04) is not implemented, and the registry/active-model card show the real metrics of all six models, generated from `results/*.json` by `scripts/export_frontend_results.py` into `frontend/model_results.js`. The old stored model list is deleted from browser storage on load. Other misleading texts were corrected: "Computing SHAP explanations" (no SHAP is computed), "Confirmed phishing threat", "Anthropic API Key Required", "results will be simulated". The fallback heuristic no longer lists rules that did not fire (they had random scores, and could be cited as reasons for a "legitimate" verdict); its score and verdict are unchanged (JS/Python parity re-checked: 0 mismatches).

**Browser tests**
- **Local server** (`python -m http.server`, built-in browser): badge "BERT backend online · phishguard-bert-base · MPS"; the app's "Phishing email" sample → phishing, >99.9%, 760 ms, 5 indicators with evidence, word chips; confidence shows 99.9% (backend now rounds down, so never 100%).
- **Script-injection check:** URL `http://<img src=x onerror="window.__xss=true">.login-secure.tk/verify` → the payload appears as plain text in the indicator evidence; 0 `<img>` elements in the result, history or dashboard; the script never ran.
- **`file://`** (headless Chrome via `scripts/browser_test.mjs`, throw-away profile): page loads from disk, both scripts load, badge online, phishing email and GitHub URL both analysed by BERT (phishing / legitimate). The built-in browser pane only shows `file://` pages as static snapshots, so it could not run this test.
- **Backend stopped** (`file://`): badge "BERT backend offline · using fallback engine"; both analyses fall back to the heuristic with the note "BERT backend offline — fallback engine used."; probability shown as "not available" (the heuristic's score is not a probability).

---

## Phase 7 — Evaluation assets, report draft, cleanup (2026-11-10)

- **Figures** (`scripts/make_figures.py`, replacing the crashing `day6_evaluation.py`): `results/figures/comparison_table.{md,csv}` (all six models, same test set), `confusion_matrices.png`, `roc_curves.png` (full and zoom), `latency.png`, `scheme_artefact.png`. Colours were checked with the dataviz palette validator (all checks pass; the contrast warning is handled with a legend plus direct labels or a table). Each figure was inspected; one label collision was found and fixed.
- **URL-scheme counterfactual** (`scripts/scheme_artefact.py` → `results/scheme_artefact.json`): every test URL without a scheme is scored as-is and with `https://` added. All five learned models flag more legitimate URLs with `https://`; the Random Forest on hand-crafted features is by far the most affected (see `results/report_draft.md`, section 5). This is the main new limitation: test-set false-positive rates understate the false alarms for real, full URLs.
- **Error analysis** (`scripts/error_analysis.py` → `results/error_analysis.json`): bert-base false positives/negatives counted by type and compared with the author's model; 10 FP and 10 FN drawn at random (seed 42). The per-example notes (`results/error_analysis_notes.json`) are the AI assistant's interpretation after reading each example — **Kasim should review them**. Recurring themes: genuine login/payment pages that look like phishing replicas, label noise (spam counted as phishing), redirect-only pages, URL-encoded lure words, and the scheme artefact.
- **Report draft** (`scripts/make_report.py` → `results/report_draft.md`): methods, results, explainability, error analysis, limitations and the "Use of third-party work" paragraph. Every number is read from `results/` at generation time (the script contains none; it prints "NOT YET MEASURED" if a file is missing). Sentences that make a claim (which models meet the target, which models are hurt by `https://`, the weakest sample type, whether the best checkpoint was the last one) are computed too. It is a draft for Kasim to rewrite in his own words.
- **README**: setup, running the backend and web app, a command for every results file, the tag timeline, and attribution. It contains no metric values on purpose.
- **Cleanup:** `PhishGuard_BERT/` removed in its own commit after checking that nothing referenced it. Every part had been replaced: `setup.bat` → `requirements.txt`; `prepare_dataset.py` → `src/data.py`; `day2_quick_test.py` → `scripts/eval_pretrained_bert.py`; `day3_finetune.py` → `scripts/train_bert.py`; `day6_evaluation.py` → `src/evaluate.py` + `scripts/make_figures.py`; `day7_demo_and_report.py` → `scripts/make_report.py`; its backend and connector were moved in Phase 1 and rewritten in Phases 5–6. The files remain in git history (e.g. `git show 48e2bed:PhishGuard_BERT/setup.bat`). After the removal, the tests, the parity check and every generator were re-run, and no committed result changed.
- The out-of-distribution test set was first left out (first item in the cut order) and then added at Kasim's request — see the next section.

### Definition of done — status

- [x] Public GitHub repo (Kasim chose public instead of the recommended private); every phase has commits and an annotated tag, all pushed; working tree clean.
- [x] No secrets, no model weights, no file over 50 MB in history (checked before every commit).
- [x] `scripts/check_env.py` shows the GPU detected — as **MPS** (Apple M4 Pro); there is no CUDA on this machine.
- [x] `results/split_manifest.json` exists; leakage handling documented (Phase 2, report draft §1 and §6).
- [x] Five models with results on the same test set, plus the pre-trained BERT evaluated on its own held-out set with the author's checkpoint-selection caveat.
- [x] Fine-tuned BERT saved locally (`models/phishguard-bert-base/`); training history and full config committed.
- [x] Backend runs, tests pass, real coverage in `results/coverage.json`.
- [x] `frontend/PhishGuard_AI.html` uses BERT with a working fallback; indicators and tokens are real.
- [x] All figures and `report_draft.md` generated from results files, no hard-coded numbers.
- [x] `README.md`, `NOTICE`, `AI_ASSISTANCE_LOG.md`, `PROGRESS.md` up to date.
- [x] Out-of-distribution test set — added after Phase 7 at Kasim's request (see below).
- [ ] Proposal requirements outside this build (FR-02, FR-04, NFR-01/02/04 load test/05/06/07/08/10, OpenAPI file) — listed in the report draft §6.

---

## Out-of-distribution evaluation (2026-11-11, at Kasim's request)

- **Data** (`scripts/build_ood.py`, run with `python -I`; files in git-ignored `data/ood/`, one folder per source): phishing = Jose Nazario's phishing corpus 2023 + 2024 (CC BY 4.0, attribution in NOTICE); legitimate = SpamAssassin public corpus easy_ham + hard_ham (2003). Text = subject line + text/plain body (or HTML body with tags removed). After dropping duplicates: **776 phishing, 2,720 legitimate emails; 0 overlap** with the original dataset (normalised exact match against train/val/test). Email text is never committed; `results/ood_manifest.json` records URLs, file hashes, licences and counts.
- **Evaluation** (`scripts/eval_ood.py <model>`, models unchanged, threshold 0.5, one process per model): results in `results/ood_<model>.json`, figure `results/figures/ood_comparison.png`, report draft section 5b with 95% Wilson intervals.
- **Bias of this set (reported with the results):** each class comes from a single source, and the legitimate emails are about 20 years older than the phishing ones, so mixed metrics such as accuracy could be inflated by "era" cues. Results are therefore read per class: recall on the phishing emails, false-positive rate on the legitimate ones.
- **Main finding** (numbers in `results/ood_bert_base.json` and report §5b): on modern phishing from another source, Kasim's bert-base catches clearly fewer phishing emails than on the original test set, while flagging very few legitimate emails. Recall drops for every learned model, most of all for the Random Forest on hand-crafted features. Differences between the two BERT models are within their confidence intervals.

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
