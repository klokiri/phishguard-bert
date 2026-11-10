"""
Evaluation figures and the model comparison table, generated ONLY from results/.

Run:  .venv/bin/python scripts/make_figures.py
Writes results/figures/:
  comparison_table.csv / comparison_table.md   every model, same test set
  confusion_matrices.png                        one panel per model
  roc_curves.png                                overlaid ROC curves (full + zoom)
  latency.png                                   mean ms per sample (batched)
  scheme_artefact.png                           false positives with/without https://

Replaces PhishGuard_BERT/notebooks/day6_evaluation.py, which crashed
(f1_score(weighted=True)) and evaluated only one model.

Colours: one fixed colour per model (validated categorical palette, never
re-ordered by rank), so a model has the same colour in every figure.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "figures"

# (results id, display name, colour, line style). Order = fixed identity order.
MODELS = [
    ("bert_base", "bert-base (Kasim, fine-tuned)", "#2a78d6", "-"),
    ("bert_pretrained", "bert-large (author) †", "#eb6834", "--"),
    ("tfidf_logreg", "TF-IDF + LogReg", "#1baf7a", "-."),
    ("features_rf", "Random Forest (features)", "#eda100", ":"),
    ("tfidf_xgboost", "TF-IDF + XGBoost", "#e87ba4", (0, (5, 1, 1, 1))),
    ("heuristic", "Heuristic (original app)", "#008300", (0, (1, 3))),
]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
FOOTNOTE = ("† author's model: not trained on this test set, but the author used it to pick the best epoch "
            "(slightly optimistic).")


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def pct(share: float) -> str:
    """Percentage with 1 decimal, but never a rounded 0.0% / 100.0% that is not exact."""
    if 0 < share < 0.0005:
        return "<0.1%"
    if 0.9995 <= share < 1:
        return ">99.9%"
    return f"{100 * share:.1f}%"


def load():
    results = {m: json.loads((RESULTS / f"{m}.json").read_text()) for m, *_ in MODELS}
    preds = {m: pd.read_csv(RESULTS / "predictions" / f"{m}.csv.gz") for m, *_ in MODELS}
    return results, preds


def comparison_table(results):
    rows = []
    for m, name, *_ in MODELS:
        r = results[m]
        rows.append({"model": name, "results_file": f"results/{m}.json", "test_rows": r["n"],
                     "accuracy": r["accuracy"], "precision": r["precision"], "recall": r["recall"],
                     "f1": r["f1"], "fpr": r["fpr"], "roc_auc": r["roc_auc"],
                     "latency_ms_per_sample": r["latency_ms_per_sample"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "comparison_table.csv", index=False)
    cols = ["model", "accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "latency_ms_per_sample"]
    md = ["| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | Latency (ms/sample) |",
          "|---|---|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
    md += ["", f"All models evaluated on the same {df['test_rows'].iloc[0]:,}-row test set. "
           "Precision/recall/F1 are for the phishing class. Latency is batched throughput on an Apple M4 Pro "
           "(BERT on the GPU via MPS, others on the CPU).", "", FOOTNOTE]
    (OUT / "comparison_table.md").write_text("\n".join(md) + "\n")
    return df


def confusion_matrices(results):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), facecolor=SURFACE)
    for ax, (m, name, *_ ) in zip(axes.flat, MODELS):
        cm = results[m]["confusion_matrix"]
        counts = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        share = counts / counts.sum(axis=1, keepdims=True)  # share of each TRUE class
        ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ink = "white" if share[i, j] > 0.6 else INK
                ax.text(j, i, f"{counts[i, j]:,}\n{pct(share[i, j])}", ha="center", va="center",
                        color=ink, fontsize=10)
        ax.set_xticks([0, 1], ["legitimate", "phishing"])
        ax.set_yticks([0, 1], ["legitimate", "phishing"])
        ax.set_xlabel("Predicted", color=INK2, fontsize=9)
        ax.set_ylabel("True", color=INK2, fontsize=9)
        ax.set_title(name, color=INK, fontsize=10.5, loc="left")
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices on the shared test set (count and % of the true class)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, FOOTNOTE, color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT / "confusion_matrices.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def roc_curves(results, preds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), facecolor=SURFACE)
    for ax, xmax in zip(axes, (1.0, 0.1)):
        style(ax)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
        for m, name, colour, ls in MODELS:
            p = preds[m]
            fpr, tpr, _ = roc_curve(p["label"], p["score"])
            ax.plot(fpr, tpr, color=colour, linestyle=ls, linewidth=2,
                    label=f"{name}  (AUC {results[m]['roc_auc']:.4f})")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0 if xmax == 1 else 0.5, 1.005)
        ax.set_xlabel("False positive rate", color=INK2, fontsize=9)
        ax.set_ylabel("True positive rate (recall)", color=INK2, fontsize=9)
        ax.set_title("ROC curves, all models" if xmax == 1 else "Zoom: false positive rate 0–0.10",
                     color=INK, fontsize=10.5, loc="left")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)
    fig.text(0.01, 0.005, FOOTNOTE, color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "roc_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def latency(results):
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    names = [name for _, name, *_ in MODELS][::-1]
    values = [results[m]["latency_ms_per_sample"] for m, *_ in MODELS][::-1]
    bars = ax.barh(names, values, color="#2a78d6", height=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f"{v:.2f} ms",
                va="center", color=INK, fontsize=9)
    ax.set_xlabel("Mean milliseconds per sample (batched, whole test set)", color=INK2, fontsize=9)
    ax.set_title("Prediction latency: Apple M4 Pro (BERT on GPU via MPS, others on CPU)",
                 color=INK, fontsize=10.5, loc="left")
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def scheme_artefact():
    data = json.loads((RESULTS / "scheme_artefact.json").read_text())
    models = [(m, name) for m, name, *_ in MODELS if m in data]
    plain = [100 * data[m]["benign"]["flagged_as_phishing_plain"] for m, _ in models]
    https = [100 * data[m]["benign"]["flagged_as_phishing_with_https"] for m, _ in models]
    n = data[models[0][0]]["benign"]["urls"]

    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(models))
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, plain, w, color="#2a78d6", label="URL as in dataset (no scheme)")
    b2 = ax.bar(x + w / 2 + 0.01, https, w, color="#eb6834", label="same URL with https:// added")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{bar.get_height():.1f}%",
                    ha="center", color=INK, fontsize=8.5)
    # Two-line labels: long model names collided side by side.
    ax.set_xticks(x, [name.replace(" (", "\n(") for _, name in models], fontsize=8.5, color=INK2)
    ax.set_ylabel("Legitimate URLs flagged as phishing (%)", color=INK2, fontsize=9)
    ax.set_title(f"Effect of adding https:// to {n:,} legitimate test URLs (nothing else changed)",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "scheme_artefact.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results, preds = load()
    df = comparison_table(results)
    confusion_matrices(results)
    roc_curves(results, preds)
    latency(results)
    scheme_artefact()
    print(df[["model", "accuracy", "f1", "fpr", "roc_auc", "latency_ms_per_sample"]].to_string(index=False))
    print("figures:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
