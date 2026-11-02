# PhishGuard AI — BERT edition

Final-year capstone by Kasim: an explainable phishing detector for emails and URLs, with a BERT model, classical baselines, a Flask backend and the PhishGuard AI web front end.

> Work in progress. See [PROGRESS.md](PROGRESS.md) for status and [AI_ASSISTANCE_LOG.md](AI_ASSISTANCE_LOG.md) for AI-assistance declarations. Setup and reproduction steps are added as each phase lands. No results have been measured yet.

## Third-party work

This project builds on **Phishing Detection with BERT** by **Esteban Alvarado**, licensed under the Apache License 2.0:

- Code: https://github.com/EstAlvB/Phishing-Detection-with-BERT — cloned locally into `reference/` (not committed) at commit `0b0b1a8c86e141bec6a4a1b7afab9fe796cd4b73`
- Dataset: https://huggingface.co/datasets/ealvaradob/phishing-dataset (revision `55caf9af01ae498d68ff44886ff69a4b165eb54b`)
- Model: https://huggingface.co/ealvaradob/bert-finetuned-phishing (revision `fa8fb73a007174c410ab7160d4e4c6e6b8d998d4`)

To recreate the local reference copy:

```bash
git clone https://github.com/EstAlvB/Phishing-Detection-with-BERT reference
git -C reference checkout 0b0b1a8c86e141bec6a4a1b7afab9fe796cd4b73
```
