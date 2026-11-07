"""
Classical (non-deep-learning) models for PhishGuard.

Three models, following the reference project's phishing-classifiers.ipynb
where it applies:

  tfidf_logreg   TF-IDF word/bigram features + Logistic Regression.
                 A simple, fast, well-understood linear baseline.
  tfidf_xgboost  The same TF-IDF features + XGBoost (gradient-boosted trees)
                 with the author's exact hyperparameters.
  features_rf    Random Forest on hand-crafted features from the reference
                 features.py (URL, text and HTML properties such as length,
                 entropy, digit counts, '@' signs, script tags). Unlike TF-IDF,
                 these features are human-readable, which suits the project's
                 explainability goal.

The TF-IDF settings are the author's: word 1-2-grams, English stop words
removed, words in more than 25% of documents ignored (max_df=0.25), the 50,000
most frequent terms kept. The vectorizer is fitted on the TRAINING data only,
so nothing about the test set leaks into the features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import HTMLFeatures, TextFeatures, URLFeatures

SEED = 42


def tfidf() -> TfidfVectorizer:
    """The author's TF-IDF configuration (phishing-classifiers.ipynb)."""
    return TfidfVectorizer(ngram_range=(1, 2), max_df=0.25, stop_words="english", max_features=50_000)


def make_tfidf_logreg() -> Pipeline:
    # A Pipeline bundles vectorizer + classifier, so the saved model takes raw
    # text in and gives probabilities out; no step can be forgotten later.
    return Pipeline([
        ("tfidf", tfidf()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def make_tfidf_xgboost() -> Pipeline:
    # Hyperparameters copied from the author's notebook.
    return Pipeline([
        ("tfidf", tfidf()),
        ("clf", XGBClassifier(
            colsample_bytree=0.7, gamma=0.2, learning_rate=0.1, max_depth=12,
            min_child_weight=2, n_estimators=100, subsample=0.8,
            objective="binary:logistic", random_state=SEED, n_jobs=-1, tree_method="hist",
        )),
    ])


def make_features_rf() -> RandomForestClassifier:
    # 300 trees; other settings are scikit-learn defaults. n_jobs=-1 uses all CPU cores.
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)


# ── Hand-crafted features ───────────────────────────────────────────────────

# Feature names, fixed order, so the table always has the same columns.
_URL_KEYS = list(URLFeatures("http://example.com").get_features().keys())
_TEXT_KEYS = list(TextFeatures("example").get_features().keys())
_HTML_KEYS = list(HTMLFeatures("<html><body>x</body></html>").get_features().keys())
FEATURE_NAMES = (
    [f"url_{k}" for k in _URL_KEYS]
    + [f"text_{k}" for k in _TEXT_KEYS]
    + [f"html_{k}" for k in _HTML_KEYS]
    + ["is_url", "is_text", "is_html", "html_parse_failed"]
)


def extract_features(text: str, type_guess: str) -> list:
    """One row of numbers describing a sample.

    URL and text features are computed for every sample (they are cheap and
    defined for any string). HTML features only make sense for web pages, so
    they are computed for HTML samples only and set to 0 otherwise; the
    is_html flag tells the model which case it is looking at.
    """
    url = URLFeatures(text).get_features()
    txt = TextFeatures(text).get_features()
    html, failed = dict.fromkeys(_HTML_KEYS, 0), 0
    if type_guess == "html":
        try:
            html = HTMLFeatures(text).get_features()
        except Exception:  # malformed markup the parser cannot handle
            failed = 1
    return (
        [float(url[k]) for k in _URL_KEYS]
        + [float(txt[k]) for k in _TEXT_KEYS]
        + [float(html[k]) for k in _HTML_KEYS]
        + [float(type_guess == "url"), float(type_guess == "text"), float(type_guess == "html"), float(failed)]
    )


def feature_table(texts, type_guesses) -> pd.DataFrame:
    rows = [extract_features(t, g) for t, g in zip(texts, type_guesses)]
    return pd.DataFrame(np.array(rows, dtype=float), columns=FEATURE_NAMES)
