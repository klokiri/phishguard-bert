"""
Train and evaluate the classical models on the shared splits.

Run:  .venv/bin/python scripts/train_classical.py [model ...]
      models: tfidf_logreg tfidf_xgboost features_rf   (default: all three)

For each model it:
  1. fits on data/splits/train.parquet only;
  2. reports validation metrics (a sanity check; no hyperparameter tuning is
     done, the settings are fixed in src/models/classical.py);
  3. evaluates on the test set through src/evaluate.py, timing prediction from
     RAW TEXT (so feature extraction counts towards latency, as it would in
     the real app);
  4. saves the fitted model to models/classical/<name>.joblib (git-ignored;
     re-create it by re-running this script).
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import compute_metrics, evaluate, load_test, print_summary  # noqa: E402
from src.models import classical  # noqa: E402

SPLITS = REPO_ROOT / "data" / "splits"
MODEL_DIR = REPO_ROOT / "models" / "classical"


def run_text_model(name, make, train, val, test):
    """TF-IDF pipelines: raw text in, probability out."""
    model = make()
    t0 = time.perf_counter()
    model.fit(train["text"], train["label"])
    train_s = time.perf_counter() - t0

    val_scores = model.predict_proba(val["text"])[:, 1]
    t0 = time.perf_counter()
    test_scores = model.predict_proba(test["text"])[:, 1]
    test_s = time.perf_counter() - t0
    return model, train_s, val_scores, test_scores, test_s


def run_features_rf(train, val, test):
    """Random Forest: feature extraction + forest. Latency includes extraction."""
    t0 = time.perf_counter()
    x_train = classical.feature_table(train["text"], train["type_guess"])
    feat_train_s = time.perf_counter() - t0
    model = classical.make_features_rf()
    t0 = time.perf_counter()
    model.fit(x_train, train["label"])
    train_s = feat_train_s + (time.perf_counter() - t0)

    val_scores = model.predict_proba(classical.feature_table(val["text"], val["type_guess"]))[:, 1]
    t0 = time.perf_counter()
    test_scores = model.predict_proba(classical.feature_table(test["text"], test["type_guess"]))[:, 1]
    test_s = time.perf_counter() - t0

    importances = sorted(zip(classical.FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    extra = {
        "html_parse_failures_train": int(x_train["html_parse_failed"].sum()),
        "top_10_feature_importances": {k: round(float(v), 4) for k, v in importances[:10]},
    }
    return model, train_s, val_scores, test_scores, test_s, extra


DESCRIPTIONS = {
    "tfidf_logreg": "TF-IDF (1-2 grams, max_df 0.25, English stop words, 50k features; author's settings) "
                    "+ LogisticRegression(max_iter=2000)",
    "tfidf_xgboost": "TF-IDF (author's settings) + XGBoost with the author's hyperparameters "
                     "(colsample_bytree 0.7, gamma 0.2, lr 0.1, max_depth 12, min_child_weight 2, "
                     "n_estimators 100, subsample 0.8)",
    "features_rf": "RandomForest(300 trees) on reference features.py: URLFeatures + TextFeatures for all "
                   "samples, HTMLFeatures for HTML samples (0 otherwise), plus type flags",
}


def main(names) -> None:
    train = pd.read_parquet(SPLITS / "train.parquet")
    val = pd.read_parquet(SPLITS / "val.parquet")
    test = load_test()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name in names:
        print(f"\n=== {name}: training on {len(train):,} rows ...", flush=True)
        extra = {}
        if name == "features_rf":
            model, train_s, val_scores, test_scores, test_s, extra = run_features_rf(train, val, test)
        else:
            make = {"tfidf_logreg": classical.make_tfidf_logreg,
                    "tfidf_xgboost": classical.make_tfidf_xgboost}[name]
            model, train_s, val_scores, test_scores, test_s = run_text_model(name, make, train, val, test)

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        val_metrics = compute_metrics(val["label"], (val_scores >= 0.5).astype(int), val_scores)
        result = evaluate(
            name, test, test_scores, latency_seconds_total=test_s,
            details={
                "description": DESCRIPTIONS[name],
                "train_rows": len(train),
                "train_seconds": round(train_s, 1),
                "validation_metrics": val_metrics,
                "latency_note": "includes text vectorisation / feature extraction; batch prediction on CPU",
                "saved_model": f"models/classical/{name}.joblib (git-ignored)",
                "command": f".venv/bin/python scripts/train_classical.py {name}",
                **extra,
            },
        )
        print(f"  trained in {train_s:.1f} s; validation accuracy {val_metrics['accuracy']}")
        print_summary(result)


if __name__ == "__main__":
    main(sys.argv[1:] or ["tfidf_logreg", "tfidf_xgboost", "features_rf"])
