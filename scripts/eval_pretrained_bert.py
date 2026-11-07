"""
Phase 4a: evaluate the author's published model on the shared test set.

Run:  .venv/bin/python scripts/eval_pretrained_bert.py
Writes results/bert_pretrained.json and results/predictions/bert_pretrained.csv.gz.

Model: ealvaradob/bert-finetuned-phishing (bert-large, 336M parameters) at
revision fa8fb73a. It is downloaded to models/pretrained/ (git-ignored) if
missing.

Leakage status (see PROGRESS.md, Phase 2): the test set is test.json from the
dataset revision this model was trained against, so the model did NOT train on
it. But the author used this same test set to choose the best training epoch,
so the score is slightly optimistic. 22 test rows also have an exact duplicate
in the training data.
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import snapshot_download  # noqa: E402

from src.evaluate import evaluate, load_test, print_summary  # noqa: E402
from src.models.bert import BertClassifier  # noqa: E402

MODEL_ID = "ealvaradob/bert-finetuned-phishing"
MODEL_REVISION = "fa8fb73a007174c410ab7160d4e4c6e6b8d998d4"
MODEL_DIR = REPO_ROOT / "models" / "pretrained" / "bert-finetuned-phishing"
BATCH_SIZE = 32


def main() -> None:
    snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR,
        allow_patterns=["config.json", "pytorch_model.bin", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"],
    )
    test = load_test()

    # 1) float16 vs float32 on a fixed random subset: does half precision
    #    change any decision?
    subset = test.sample(400, random_state=0)
    p32 = BertClassifier(str(MODEL_DIR), half=False).predict_proba(subset["text"], BATCH_SIZE)
    clf = BertClassifier(str(MODEL_DIR), half=True)
    p16 = clf.predict_proba(subset["text"], BATCH_SIZE)
    precision_check = {
        "rows": len(subset),
        "max_abs_prob_difference": round(float(np.abs(p16 - p32).max()), 5),
        "decision_disagreements_at_0.5": int(((p16 >= 0.5) != (p32 >= 0.5)).sum()),
    }
    print("fp16 vs fp32:", precision_check)

    # 2) Single-request latency, as the backend would see it (batch of one),
    #    on 50 random test rows, after a warm-up.
    one_by_one = test.sample(50, random_state=1)["text"].tolist()
    clf.predict_proba(one_by_one[:3], batch_size=1)
    times = []
    for text in one_by_one:
        t0 = time.perf_counter()
        clf.predict_proba([text], batch_size=1)
        times.append(1000 * (time.perf_counter() - t0))
    single = {
        "rows": len(times),
        "mean_ms": round(float(np.mean(times)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
        "max_ms": round(float(np.max(times)), 1),
    }
    print("single-request latency:", single)

    # 3) The full test set, batched.
    t0 = time.perf_counter()
    scores = clf.predict_proba(test["text"], BATCH_SIZE, progress=True)
    elapsed = time.perf_counter() - t0

    result = evaluate(
        "bert_pretrained", test, scores, latency_seconds_total=elapsed,
        details={
            "description": f"{MODEL_ID} @ {MODEL_REVISION} (Esteban Alvarado; bert-large-uncased "
                           "fine-tuned, 336M parameters). Inference only, no training by us.",
            "independence": "Test set = held-out test.json of the dataset revision the model was trained "
                            "on (not in its training data). The author used this test set for per-epoch "
                            "evaluation and best-checkpoint selection, so the score is slightly optimistic. "
                            "22 test rows have an exact duplicate in train.json.",
            "author_reported_on_same_test_set": {
                "source": "HF model card, epoch 4", "accuracy": 0.9717, "precision": 0.9658,
                "recall": 0.9670, "false_positive_rate": 0.0249,
            },
            "inference": f"device {clf.device.type}, float16={clf.half}, batch {BATCH_SIZE}, "
                         f"truncation to {clf.max_length} tokens, length-sorted batches",
            "fp16_vs_fp32_check": precision_check,
            "single_request_latency": single,
            "latency_note": "latency_ms_per_sample is batched throughput; see single_request_latency "
                            "for one request at a time",
            "command": ".venv/bin/python scripts/eval_pretrained_bert.py",
        },
    )
    print_summary(result)


if __name__ == "__main__":
    main()
