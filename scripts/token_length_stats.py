"""
Token-length statistics of the training data, to choose BERT's max_length.

Run:  .venv/bin/python scripts/token_length_stats.py
Writes results/token_length_stats.json.

Why it matters: BERT's cost grows faster than linearly with sequence length,
and on this Mac training time is the main constraint. If most samples are
short, a smaller max_length (e.g. 256) loses little information but trains
much faster. We measure instead of guessing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = "bert-base-uncased"
CANDIDATES = [64, 128, 256, 384, 512]


def main() -> None:
    train = pd.read_parquet(REPO_ROOT / "data" / "splits" / "train.parquet", columns=["text", "type_guess"])
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    # Cap the characters looked at: no text needs more than ~10 characters per
    # token, so 20,000 characters is far beyond 512 tokens either way. This
    # keeps the one 17-million-character page from dominating the run time.
    texts = [t[:20_000] for t in train["text"]]
    lengths = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])

    def stats(mask):
        lens = lengths[mask]
        return {
            "rows": int(mask.sum()),
            "percentiles": {f"p{p}": int(np.percentile(lens, p)) for p in (50, 75, 90, 95, 99)},
            "fraction_fitting_without_truncation": {
                str(m): round(float((lens <= m).mean()), 4) for m in CANDIDATES
            },
        }

    report = {
        "tokenizer": TOKENIZER,
        "data": "data/splits/train.parquet (texts capped at 20,000 characters before tokenising)",
        "all": stats(np.ones(len(lengths), dtype=bool)),
        "by_type": {t: stats((train["type_guess"] == t).to_numpy()) for t in sorted(train["type_guess"].unique())},
        "command": ".venv/bin/python scripts/token_length_stats.py",
    }
    (REPO_ROOT / "results" / "token_length_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
