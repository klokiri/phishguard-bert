# Task: Complete PhishGuard BERT (capstone build, 7 days, tracked on GitHub)

## Who you are working with

You are helping Kasim, a final-year IT/cybersecurity student, finish his capstone project **PhishGuard AI** within 7 days. He knows Python basics (variables, loops, functions) but is new to PyTorch, Hugging Face, Flask and Git. He must be able to explain every part of this system to his supervisor, so your job is to build it **and** teach it.

He is working on his own machine with an NVIDIA GPU (probably Windows; confirm this). The whole build is version-controlled in a GitHub repository that you create in Phase 0, with a commit after every major step.

## What is in this folder (the starter bundle)

Check each item yourself; do not assume.

```
phishguard-ai/
├── PHISHGUARD_BERT_PROMPT.md               ← this file
├── PhishGuard_AI.html                      ← the current web app
├── PhishGuard_BERT/                        ← scaffold from an earlier chat; NEVER RUN
│   ├── setup.bat
│   ├── README.md
│   ├── data/prepare_dataset.py
│   ├── notebooks/day2_quick_test.py, day3_finetune.py, day6_evaluation.py, day7_demo_and_report.py
│   ├── backend/app.py                      (Flask + SQLite)
│   └── frontend-patch/bert_connector.js
└── docs/PhishGuardAI_Project_Proposal.docx ← the capstone proposal
```

- **`PhishGuard_AI.html`** is a single-file web app (vanilla JS, Chart.js, localStorage).
  - Its analysis engine calls either the Claude API or a regex heuristic, `simulateAnalysis()`.
  - The result object shape the UI expects is `{verdict, confidence, indicators: [{name, score, direction}], explanation}`.
  - The strings `sk-ant-api03-...` and `startsWith('sk-ant-')` in it are input placeholders and validation code, not real keys.
- **`PhishGuard_BERT/`** is a draft. Treat it as a draft, not working code.
- **The proposal** defines:
  - functional requirements FR-01 to FR-04 and non-functional requirements NFR-01 to NFR-12;
  - an objective of at least 3 ML models, with ≥95% accuracy and F1 ≥ 0.93;
  - explainable outputs as the main selling point.

  Read it in Phase 0. A `.docx` is a zip archive of XML, so you can extract the text with `python-docx` or by unzipping it.

## The reference project (third-party work)

Kasim is building on **Esteban Alvarado's** final-year project, licensed under Apache-2.0.

**Repo:** https://github.com/EstAlvB/Phishing-Detection-with-BERT

- Notebooks: `preprocessing.ipynb`, `analysis.ipynb`, `phishing-classifiers.ipynb`, `finetuning-bert.ipynb`.
- `features.py` contains:
  - `URLFeatures` (19 features)
  - `TextFeatures` (6 features)
  - `HTMLFeatures` (23 features)
- `features.py` imports `pyquery` at module level, so install `pyquery` even if you only use the URL and text features.

**Model:** `ealvaradob/bert-finetuned-phishing`

- A fine-tune of `bert-large-uncased`: 336M parameters, about 1.3 GB, 2 labels.
- Trained on URLs, emails, SMS messages and website HTML.
- Training settings: lr 2e-5, batch size 16, 4 epochs, seed 42.
- Author-reported evaluation results: accuracy 0.9717, precision 0.9658, recall 0.9670, false positive rate 0.0249.

**Dataset:** `ealvaradob/phishing-dataset`

- Columns are `text` and `label`, where 1 = phishing and 0 = benign.
- It uses a loading script:
  ```python
  load_dataset("ealvaradob/phishing-dataset", "combined_reduced", trust_remote_code=True)
  ```
- Only a `train` split exists.
- `combined_reduced` is the config the author recommends and almost certainly the one the model was trained on. Confirm this in the notebook.
- Read the loading script for the other config names rather than guessing them.
- Recent `datasets` releases (4.x) no longer run loading scripts. Use one of these approaches:
  - pin `datasets<4`, or
  - download the underlying data files with `huggingface_hub` and load them with pandas.
- Download only the files you need. The full repo is about 1.7 GB.

**Before writing any code:** clone the repo into `reference/` (this folder is git-ignored; record the commit hash you cloned in `README.md`). Read all four notebooks plus `features.py`, then record these points in `PROGRESS.md`:

- exactly how the author split the data (test size, random seed, stratification);
- his preprocessing steps;
- which classical classifiers he compared and how.

## Critical: data leakage

The pre-trained model has already seen most of `combined_reduced` during training. If you evaluate it on a random split you create yourself, its test set will contain its own training data and the results will be inflated. In a capstone, that is a fatal methodological flaw. Handle it as follows:

1. Find the exact split in `finetuning-bert.ipynb` and reproduce it. Evaluate the pre-trained model **only** on that held-out portion.
2. Use that same train/test split for every model Kasim trains, so that all comparisons are made on one identical test set.
3. If the split cannot be reproduced exactly:
   - say so plainly in `PROGRESS.md` and in the report draft;
   - present the pre-trained model's figures as "author-reported" only;
   - do not present your own evaluation of that model on this dataset as independent.
4. If time allows, build a small out-of-distribution test set from a different public phishing/benign email source that is not part of this dataset. Report it separately and record its source.

## Known defects in the existing scaffold (verify, then fix)

- **`prepare_dataset.py`**
  - Loads the `emails` config without `trust_remote_code=True`.
  - Its random 70/15/15 split ignores the leakage issue above.
  - It truncates every text to 1000 characters without justification.
- **`day2_quick_test.py` and `README.md`** describe the model as about 440 MB. It is bert-large, about 1.3 GB.
- **`day6_evaluation.py`**
  - `f1_score(..., weighted=True)` raises a `TypeError`; the parameter is `average=`.
  - Decide whether F1 is reported for the phishing class (binary) or as a weighted average, document the choice, and use it consistently everywhere.
- **`backend/app.py`**
  - `_build_indicators()` adds `np.random` noise and invents scores for signals that did not fire. Indicators must be deterministic and truthful.
  - The "suspicious" band logic is inconsistent with the 0.4/0.6 thresholds in `day7`. Put all thresholds in one config.
- **`day7_demo_and_report.py`**
  - Falls back to hard-coded metrics (0.943 and others) when results are missing.
  - States baseline accuracies (~74%, ~89%) that were never measured.
  - Claims "attention-weight-derived indicators" that do not exist.
  - Maps NFR-09 (test coverage) to model accuracy.
  - All of this must be removed.
- **`setup.bat`** pins CUDA 11.8 wheels and specific library versions without checking the GPU or driver. Run `nvidia-smi` first and choose the matching PyTorch build.
- **`bert_connector.js`**
  - Check that it works when the HTML is opened via `file://` and via a local server.
  - Integrate it properly with the existing `onclick="runAnalysis()"` instead of overriding handlers after the fact.

## Git and GitHub workflow

### Repo setup (part of Phase 0)

1. **Check the tools.** Run `git --version`, `gh --version` and `gh auth status`.
   - If the GitHub CLI is missing, give Kasim the install command (`winget install --id GitHub.cli` on Windows).
   - If it is not authenticated, have Kasim run `gh auth login` himself in his own terminal and browser.
   - **Never ask for, store, print or commit tokens or passwords.**
2. **Check the identity.** Make sure `git config user.name` and `git config user.email` are set. If not, ask Kasim which name and email to use; suggest his GitHub noreply address if he prefers privacy.
3. **Stop and ask Kasim to confirm:**
   - **Repo name.** Suggest `phishguard-ai`.
   - **Visibility.** Recommend **private** until the capstone is graded, because a public repo lets other students copy assessed work. He can make it public afterwards for his portfolio.
4. **Before the first commit:**
   - create `.gitignore` and `.gitattributes` (contents below);
   - scan every file for secrets (real-looking `sk-ant-…` keys, `ghp_…` tokens, `.env` files, passwords). The placeholders noted above are expected. Stop if anything real turns up.
5. **Create the baseline commit.** Run `git init -b main`, then commit the bundle **exactly as received**:
   ```
   chore: import starting files (PhishGuard_AI.html, unrun BERT scaffold, proposal)
   ```
   This baseline makes every later change visible in the history.
6. **Create the GitHub repo and push:**
   ```
   gh repo create <name> --private --source=. --remote=origin --push
   ```
   Write the repo URL in `PROGRESS.md` and show it to Kasim.
   - **If `gh` cannot be used:** Kasim creates an empty repo on github.com (no README, no .gitignore) and gives you the URL. You then run `git remote add origin <url>` and `git push -u origin main`.

### `.gitignore` must cover at least

- `.venv/`
- `__pycache__/` and `*.pyc`
- `.ipynb_checkpoints/`
- `data/` (raw data and split files)
- `models/`, `checkpoints/`, `runs/`, `wandb/`
- `reference/`
- `*.db`
- `.env`
- `*.log`
- Hugging Face cache folders

`results/` is **not** ignored: its JSON files and figures are committed as evidence. Keep the split manifest in `results/split_manifest.json`.

### `.gitattributes`

```
* text=auto
*.bat text eol=crlf
*.png binary
*.docx binary
```

### Commit rules

**When to commit**

- Commit at every step in the commit plan below, and at any other meaningful sub-step.
- Commit **only after that step's code has actually run successfully**, and after its tests pass where tests exist. Never commit code you have not run.
- Push after every commit, then tell Kasim the short hash and a one-line summary.

**Message format**

Use Conventional Commits:

```
type(scope): summary
```

- **Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `data`, `exp` (experiment results).
- **Body:** what changed, why, and how it was verified (the exact command run and its key output).

**What goes into each commit**

- Commit results (`results/*.json`, `results/figures/*`) **together with** the code that produced them, and put the exact command that produced them in the commit body. This is the reproducibility evidence for the capstone.
- Include the updated `PROGRESS.md` and `AI_ASSISTANCE_LOG.md` in each phase-closing commit.
- Keep Claude Code's AI co-author trailer on every commit you make; together with `AI_ASSISTANCE_LOG.md`, it documents AI assistance.
- If Kasim edits code himself, he commits it himself in a separate commit, so his own contributions stay distinguishable.

**Checks before every commit**

- Run `git status` and `git diff --staged --stat`.
- Stage specific paths; do not run `git add -A` blindly.
- Make sure no staged file is over 50 MB, nothing from ignored folders is staged, and there are no secrets.

**Moving files and tagging**

- Move files with `git mv` so their history is preserved.
- At the end of each phase, create an annotated tag and push it:
  ```
  git tag -a phase-<N>-<name> -m "<summary>"
  git push origin --tags
  ```

**Model weights**

- Never commit model weights: GitHub rejects files over 100 MB and the weights can be regenerated.
- Document how to regenerate them. Upload to a private Hugging Face repo only if Kasim asks.

**Never do any of the following without asking first:**

- force-push;
- amend or rewrite pushed history;
- `git reset --hard`;
- delete branches;
- change repo settings or visibility.

### Commit plan (minimum)

| Phase | Commits |
|---|---|
| 0 | `chore: import starting files…` · `docs: add PROGRESS.md with recon findings and 7-day schedule` |
| 1 | `chore: restructure repo layout` (`git mv` only, no content changes) · `chore(env): add requirements, gitignore, check_env script` |
| 2 | `data: load combined_reduced, reproduce author split, add manifest` |
| 3 | `feat(eval): shared evaluate.py` · `feat(baseline): port JS heuristic to Python + results` · `feat(baseline): classical models + results` |
| 4 | `exp(bert): evaluate pre-trained model on held-out split` · `feat(bert): fine-tune bert-base + training history` |
| 5 | `refactor(backend): central config and ModelService interface` · `feat(backend): deterministic indicators and token attribution` · `test(backend): API tests and coverage report` |
| 6 | `feat(frontend): connect PhishGuard_AI.html to BERT backend` |
| 7 | `exp: model comparison figures and error analysis` · `docs: report draft, README, attribution` · `chore: remove superseded PhishGuard_BERT scaffold` |

## Target repo layout (after Phase 1)

```
phishguard-ai/
├── README.md  PROGRESS.md  AI_ASSISTANCE_LOG.md  NOTICE  PHISHGUARD_BERT_PROMPT.md
├── requirements.txt  .gitignore  .gitattributes
├── frontend/      PhishGuard_AI.html, bert_connector.js
├── backend/       app.py and modules
├── src/           data.py, evaluate.py, features.py (adapted from reference), models/
├── scripts/       check_env.py, train_bert.py, …
├── tests/
├── results/       *.json, split_manifest.json, figures/      (committed)
├── docs/          proposal
├── third_party/   LICENSE-EstAlvB.md (Apache-2.0 text from the reference repo)
├── PhishGuard_BERT/   (old scaffold; removed in Phase 7 once fully superseded)
└── reference/  data/  models/   (git-ignored)
```

If you copy `features.py` into `src/`, add a header comment to it with:

- the source URL;
- the author (Esteban Alvarado);
- the license (Apache-2.0);
- the commit hash you copied from;
- a note on what you changed.

Also add a `NOTICE` file in the repo root crediting the reference project.

## Academic integrity rules (non-negotiable)

- **Never invent, estimate, round up or hard-code a result.**
  - Every number in any chart, table, README or report draft must come from code in this repo and be read from a results file such as `results/*.json`.
  - If something has not been measured, write `NOT YET MEASURED`.
- **Keep the boundary between authors clear.** Esteban Alvarado's work is the repo, dataset, pre-trained model and `features.py`; everything else is Kasim's.
  - Attribution and the Apache-2.0 notice go in the README and `NOTICE`.
  - Draft a short "Use of third-party work" paragraph for Kasim's report.
- **Log AI-generated work.** Maintain `AI_ASSISTANCE_LOG.md`: for each phase, list the files you generated or changed, what each does, and the commit hashes. Kasim can use this to declare AI assistance as his unit requires.
- **Never describe a feature that was not implemented.**

## Phases

Work through the phases in order. Close every phase with this routine:

1. update `PROGRESS.md` and `AI_ASSISTANCE_LOG.md`;
2. commit, push and tag (see commit rules);
3. explain to Kasim in plain language (at most 10 lines) what you built and why;
4. give one question his supervisor might ask about it, with a good answer;
5. wait for Kasim to say "continue" before starting the next phase.

### Phase 0: Recon and repo setup

- Confirm the OS, Python version, GPU model, VRAM, driver/CUDA version (`nvidia-smi`) and free disk space.
- Read the bundle, the proposal and the reference repo. Change no code in this phase.
- Do the repo setup above. Stop for Kasim's name and visibility choice before creating the GitHub repo.
- Write `PROGRESS.md` with your findings, the repo URL and a realistic day-by-day schedule that fits 7 days.

### Phase 1: Restructure and environment

- **Restructure:** move the existing files into the target layout with `git mv`, and commit that move on its own.
- **Environment:**
  - create a virtual environment in `.venv`;
  - write `requirements.txt` with versions you have verified install together, and install the PyTorch build that matches his CUDA version;
  - write `scripts/check_env.py`, which prints library versions and `torch.cuda.is_available()`.

### Phase 2: Data

Write `src/data.py`. It should:

- load `combined_reduced`;
- reproduce the author's split;
- carve a validation set from the train portion only;
- save `data/splits/{train,val,test}.parquet` (git-ignored);
- write the SHA-256 hash of each split file to `results/split_manifest.json` (committed);
- write `results/data_summary.json` with sample counts and class balance (committed).

### Phase 3: Shared evaluation and baselines (the ≥3 models the proposal promises)

- **Shared evaluation first:** write `src/evaluate.py`. It computes identical metrics for every model and writes them to `results/<model>.json`:
  - accuracy, precision, recall, F1, ROC-AUC, false positive rate;
  - confusion matrix;
  - mean latency per sample.
- **Baseline 0:** port the regex heuristic from `simulateAnalysis()` in `PhishGuard_AI.html` to Python.
- **Classical models:** following the approach in `phishing-classifiers.ipynb`, train at least two, for example:
  - TF-IDF + Logistic Regression;
  - Random Forest on `features.py` features.
- Evaluate everything on the same test set.

### Phase 4: BERT

**4a: Evaluate the pre-trained model**

- Evaluate `ealvaradob/bert-finetuned-phishing` on the held-out test set, obeying the leakage rules above.
- Use batched inference, with fp16 if the GPU supports it.

**4b: Fine-tune Kasim's own model**

- Fine-tune `bert-base-uncased` on the train split. If VRAM is under 8 GB, use `distilbert-base-uncased` instead.
- Use the Hugging Face `Trainer` with fp16, early stopping on the validation set and seed 42.
- Choose `max_length` from token-length statistics of the data.
- Save the model to `models/phishguard-bert-base/` (git-ignored) and the training history to `results/bert_base_history.json` (committed).
- Make training resumable from checkpoints.
- Before starting, give a time estimate and ask permission.

### Phase 5: Backend

Refactor `backend/app.py`:

- **One config:** model name/path via an environment variable, thresholds, `max_length`.
- **Swappable models:** a `ModelService` interface so the heuristic, classical and BERT models can be swapped (this is NFR-12).
- **Indicators that are real and deterministic:**
  - use `URLFeatures` for URL input;
  - use `TextFeatures` plus documented keyword checks for email input;
  - each indicator has a `name`, a 0–100 `score` derived by a documented rule, and a `direction`.
- **Token attribution:** add a simple, honest method that returns the top contributing words. Occlusion (leave one word out) is enough; Captum Integrated Gradients is optional if time allows. Name the method accurately in the UI.
- **Explanation text:** build it only from signals that actually fired.
- **Keep from the scaffold:**
  - SQLite logging, using parameterised queries only;
  - rate limiting;
  - input validation.
- **Tests:** write pytest tests with the model mocked, covering:
  - a valid request returns 200;
  - empty content returns 422;
  - an invalid `input_type` returns 422;
  - exceeding the rate limit returns 429.
- **Coverage:** measure it with `pytest-cov` and save the real figure to `results/coverage.json` as the NFR-09 result.

### Phase 6: Frontend

- Git already preserves the original version, so no separate backup copy is needed. Make sure the pre-change state is committed before you edit.
- Integrate the connector so that:
  - backend status is visible in the UI;
  - BERT is the primary engine and the heuristic is the fallback;
  - the result card shows phishing probability, model name, latency and top tokens;
  - all model-derived text is inserted via `escHtml()` or `textContent`.
- Test in a browser both via `python -m http.server` and via `file://`.

### Phase 7: Evaluation assets, report draft, cleanup

- Fix and rerun the evaluation dashboard. Save the following to `results/figures/`:
  - one comparison table covering every model on the same test set;
  - confusion matrices;
  - overlaid ROC curves;
  - a latency chart.
- **Error analysis:** take 10 real false positives and 10 real false negatives from the test set and add a short note on each.
- **Report draft:** the generator reads **only** from `results/` and writes `results/report_draft.md`. It covers:
  - methods and results;
  - limitations: how leakage was handled, dataset source bias, English-only data;
  - the third-party-work paragraph.
- **README:**
  - setup steps;
  - how to reproduce every number (command per results file);
  - attribution;
  - the repo's tag list as a build timeline.
- **Cleanup:** once nothing still depends on it, remove `PhishGuard_BERT/` in its own commit (it stays in the git history).

## If time runs short

**Must-haves:**

- a clean data split;
- at least 3 models evaluated on the same test set;
- a working backend and frontend using BERT;
- honest results;
- everything committed and pushed.

**Cut in this order:**

1. the out-of-distribution test set
2. token attribution (keep the feature-based indicators)
3. the batch endpoint
4. HTML input support

## Working rules

- Work in small steps and run what you write. Never say something works unless you have run it.
- Comment code for a beginner: explain why, not just what.
- When you run a git command Kasim hasn't seen before, explain it in one line.
- **Ask before:**
  - downloads over 2 GB;
  - training runs longer than 15 minutes;
  - deleting or overwriting any of Kasim's files;
  - any of the git actions listed under "Never do any of the following without asking first".
- If something is ambiguous (for example, the author's split cannot be found), stop, lay out the options and let Kasim choose.

## Definition of done

- [ ] A private GitHub repo exists. Every phase has commits and an annotated tag, all pushed, and the working tree is clean.
- [ ] The git history contains no secrets, no model weights and no files over 50 MB.
- [ ] `scripts/check_env.py` shows the GPU detected.
- [ ] `results/split_manifest.json` exists, and the leakage handling is documented.
- [ ] At least 3 models have results in `results/` on the same test set, plus the pre-trained BERT evaluated (or flagged as author-reported).
- [ ] The fine-tuned BERT model is saved locally, and its training history is committed.
- [ ] The backend runs, tests pass, and real coverage is saved to `results/coverage.json`.
- [ ] `frontend/PhishGuard_AI.html` uses BERT with a working fallback; indicators and tokens are real.
- [ ] All figures and `report_draft.md` are generated from results files, with no hard-coded numbers.
- [ ] `README.md`, `NOTICE`, `AI_ASSISTANCE_LOG.md` and `PROGRESS.md` are up to date.

**Start with Phase 0.**
