"""
Measure every indicator rule on the TRAINING split, so indicator scores come
from data rather than from hand-picked numbers.

Run:  .venv/bin/python scripts/build_indicator_stats.py
Writes results/indicator_stats.json (read by the backend at start-up).

URL rules are measured on training samples of type "url"; email rules on
type "text" (emails and SMS). HTML pages are excluded: the app has no HTML
input mode. Only data/splits/train.parquet is used - never validation or test.
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.indicators import EXCLUDED_FROM_DISPLAY, MIN_LIFT, MIN_SUPPORT, RULES, STATS_PATH  # noqa: E402

TYPE_FOR_INPUT = {"url": "url", "email": "text"}


def main() -> None:
    train = pd.read_parquet(REPO_ROOT / "data" / "splits" / "train.parquet", columns=["text", "label", "type_guess"])
    report = {"data": "data/splits/train.parquet", "min_support": MIN_SUPPORT, "min_lift": MIN_LIFT,
              "excluded_from_display": EXCLUDED_FROM_DISPLAY,
              "base_phishing_rate": {}, "rows": {}, "rules": {}}
    for input_type, rules in RULES.items():
        part = train[train["type_guess"] == TYPE_FOR_INPUT[input_type]]
        labels = part["label"].to_numpy()
        report["base_phishing_rate"][input_type] = round(float(labels.mean()), 4)
        report["rows"][input_type] = len(part)
        for rule in rules:
            fired = [rule.check(t) is not None for t in part["text"]]
            n = sum(fired)
            phishing = int(sum(lab for lab, f in zip(labels, fired) if f))
            report["rules"][rule.id] = {
                "input_type": input_type, "name": rule.name, "fired": n,
                "phishing_rate": round(phishing / n, 4) if n else None,
            }
            rate = report["rules"][rule.id]["phishing_rate"]
            print(f"{input_type:5} {rule.id:24} fired {n:6,}  phishing rate {rate}")
    report["command"] = ".venv/bin/python scripts/build_indicator_stats.py"
    STATS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print("base rates:", report["base_phishing_rate"], "->", STATS_PATH.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
