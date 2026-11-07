"""
BERT inference wrapper, shared by evaluation scripts and the backend.

Works for any Hugging Face sequence-classification checkpoint with a
"phishing" label: the author's ealvaradob/bert-finetuned-phishing (bert-large)
and Kasim's own fine-tuned bert-base.

Why the batching code looks the way it does:
  - Texts are tokenised once, truncated to max_length tokens (BERT's limit is
    512). Longer texts keep only their first max_length tokens.
  - Batches are built from texts of similar length (sorted by token count).
    Each batch is only padded to its own longest member, so a batch of short
    URLs costs far less than padding everything to 512. Results are put back
    in the original order at the end.
  - On Apple Silicon ("mps") or CUDA the model can run in float16, which
    roughly halves memory and speeds things up. Probabilities are computed in
    float32 for accuracy. scripts/eval_pretrained_bert.py checks that float16
    and float32 give the same answers.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BertClassifier:
    def __init__(self, model_path: str, device=None, half: bool = True, max_length: int = 512):
        self.device = device or best_device()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # float16 only on a GPU; on CPU it is slow or unsupported.
        self.half = half and self.device.type in ("cuda", "mps")
        if self.half:
            self.model = self.model.half()
        self.model.to(self.device).eval()  # eval(): disable dropout for inference
        # Which output column means "phishing" (read from the model's config,
        # not assumed).
        self.phishing_index = int(self.model.config.label2id.get("phishing", 1))
        self.name = model_path

    def tokenize(self, texts):
        return self.tokenizer(list(texts), truncation=True, max_length=self.max_length)["input_ids"]

    @torch.no_grad()  # no gradients needed for prediction: saves memory and time
    def predict_proba(self, texts, batch_size: int = 32, progress: bool = False) -> np.ndarray:
        """Phishing probability for each text, in the input order."""
        ids = self.tokenize(texts)
        order = np.argsort([len(x) for x in ids])  # shortest first
        probs = np.empty(len(ids), dtype=np.float64)

        batches = range(0, len(order), batch_size)
        if progress:
            from tqdm.auto import tqdm
            batches = tqdm(batches, desc="BERT inference", unit="batch")
        for start in batches:
            idx = order[start:start + batch_size]
            batch = self.tokenizer.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch).logits.float()
            probs[idx] = torch.softmax(logits, dim=-1)[:, self.phishing_index].cpu().numpy()
        return probs
