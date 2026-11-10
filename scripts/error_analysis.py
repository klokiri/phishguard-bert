"""
Error analysis of Kasim's fine-tuned model (bert_base) on the test set.

Run:  .venv/bin/python scripts/error_analysis.py
Writes results/error_analysis.json.

What it does:
  - counts false positives (legitimate flagged as phishing) and false
    negatives (phishing missed) by sample type;
  - checks how many of those errors the author's bert-large also makes;
  - draws 10 false positives and 10 false negatives AT RANDOM (seed 42), so
    the examples are representative rather than cherry-picked, and stores a
    short excerpt of each.
The per-example notes are written separately, after reading each example, in
results/error_analysis_notes.json (see the report draft).
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import load_test  # noqa: E402

N_EXAMPLES = 10
SEED = 42
EXCERPT_CHARS = 300


def predictions(model: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "results" / "predictions" / f"{model}.csv.gz")


def main() -> None:
    test = load_test()[["row_id", "text", "type_guess", "label"]]
    ours = predictions("bert_base")[["row_id", "score", "pred"]]
    theirs = predictions("bert_pretrained")[["row_id", "pred"]].rename(columns={"pred": "pred_author"})
    df = test.merge(ours, on="row_id").merge(theirs, on="row_id")
    assert len(df) == len(test), "row_id join must keep every test row"

    fp = df[(df["label"] == 0) & (df["pred"] == 1)]
    fn = df[(df["label"] == 1) & (df["pred"] == 0)]

    def summary(errors):
        return {
            "count": len(errors),
            "by_type": {k: int(v) for k, v in errors["type_guess"].value_counts().sort_index().items()},
            "also_wrong_in_author_model": int((errors["pred_author"] == errors["pred"]).sum()),
        }

    def examples(errors):
        picked = errors.sample(n=min(N_EXAMPLES, len(errors)), random_state=SEED).sort_values("row_id")
        return [{
            "row_id": int(r.row_id),
            "type": r.type_guess,
            "true_label": "phishing" if r.label == 1 else "legitimate",
            "phishing_score": round(float(r.score), 4),
            "author_model_also_wrong": bool(r.pred_author == r.pred),
            "chars": len(r.text),
            "excerpt": re.sub(r"\s+", " ", r.text)[:EXCERPT_CHARS],
        } for r in picked.itertuples()]

    report = {
        "model": "bert_base",
        "test_rows": len(df),
        "false_positives": summary(fp),
        "false_negatives": summary(fn),
        "sampling": f"{N_EXAMPLES} of each drawn with pandas .sample(random_state={SEED})",
        "false_positive_examples": examples(fp),
        "false_negative_examples": examples(fn),
        "command": ".venv/bin/python scripts/error_analysis.py",
    }
    (REPO_ROOT / "results" / "error_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("false_positives", "false_negatives")}, indent=2))
    for kind in ("false_positive_examples", "false_negative_examples"):
        print(f"\n=== {kind}")
        for e in report[kind]:
            print(f"- row {e['row_id']} [{e['type']}, score {e['phishing_score']}, "
                  f"author also wrong: {e['author_model_also_wrong']}, {e['chars']} chars]\n  {e['excerpt'][:260]}")


if __name__ == "__main__":
    main()
