| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | Latency (ms/sample) |
|---|---|---|---|---|---|---|---|
| bert-base (Kasim, fine-tuned) | 0.9655 | 0.9631 | 0.9547 | 0.9589 | 0.0266 | 0.9935 | 6.2714 |
| bert-large (author) † | 0.9716 | 0.9656 | 0.967 | 0.9663 | 0.025 | 0.9944 | 17.7278 |
| TF-IDF + LogReg | 0.8842 | 0.906 | 0.8088 | 0.8546 | 0.061 | 0.9627 | 0.3633 |
| Random Forest (features) | 0.8735 | 0.873 | 0.8184 | 0.8448 | 0.0865 | 0.9328 | 4.0027 |
| TF-IDF + XGBoost | 0.846 | 0.9131 | 0.7008 | 0.793 | 0.0485 | 0.9393 | 0.4154 |
| Heuristic (original app) | 0.5792 | 1.0 | 0.0003 | 0.0006 | 0.0 | 0.5897 | 0.3206 |

All models evaluated on the same 15,466-row test set. Precision/recall/F1 are for the phishing class. Latency is batched throughput on an Apple M4 Pro (BERT on the GPU via MPS, others on the CPU).

† author's model: not trained on this test set, but the author used it to pick the best epoch (slightly optimistic).
