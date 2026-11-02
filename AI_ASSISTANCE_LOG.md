# AI Assistance Log

This log records the work generated or changed by an AI assistant (Claude Code, model Claude Opus 5.5) during this project, so that Kasim can declare AI assistance as his unit requires. Every AI-made commit also carries a `Co-Authored-By: Claude` trailer. Commits without that trailer are Kasim's own.

## Phase 0 — Recon and repo setup (2026-11-02)

| File | What it is | Commit |
|---|---|---|
| `.gitignore` | Ignore rules (venv, data, models, reference repo, secrets, caches); `results/` deliberately tracked | `48e2bed` |
| `.gitattributes` | Line-ending and binary rules | `48e2bed` |
| `PhishGuard_BERT/`, `docs/` | **Not AI-written content**: unpacked unchanged from `phishguard-ai-starter.zip` (scaffold from an earlier chat; proposal) | `48e2bed` |
| `PROGRESS.md` | Recon findings (machine, scaffold defects, proposal, reference analysis, leakage red flags) and 7-day schedule | see `git log -- PROGRESS.md` |
| `README.md` | Initial README with third-party attribution and pinned reference commit | see `git log -- README.md` |
| `AI_ASSISTANCE_LOG.md` | This file | see `git log -- AI_ASSISTANCE_LOG.md` |

Also done by the AI in Phase 0 (no files): checked the environment, read the four reference notebooks and `features.py`, read the proposal text, ran a secret scan, created the GitHub repo `klokiri/phishguard-bert` (public, as Kasim chose) and pushed.
