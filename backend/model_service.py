"""
ModelService: the one interface every model implements (NFR-12).

The API layer (backend/app.py) only ever calls these methods, so the
heuristic, a classical model or BERT can be swapped by changing one setting
(PHISHGUARD_MODEL) with no change to the API code. The tests prove this by
running the same API tests against a fake model and the real heuristic.

    predict_proba(texts, input_type) -> phishing probability per text (0..1)
    info()                           -> name, kind, device etc. for /health
"""

from abc import ABC, abstractmethod

import numpy as np

from src.config import Config


class ModelService(ABC):
    name: str = "model"
    kind: str = "abstract"
    device: str = "cpu"

    @abstractmethod
    def predict_proba(self, texts: list[str], input_type: str) -> np.ndarray:
        """Phishing probability for each text, same order as the input."""

    def info(self) -> dict:
        return {"name": self.name, "kind": self.kind, "device": self.device}


class BertService(ModelService):
    """A fine-tuned BERT checkpoint (Kasim's bert-base by default)."""

    kind = "bert"

    def __init__(self, config: Config):
        # Imported here so the heuristic/classical options don't need to load torch.
        from src.models.bert import BertClassifier

        self.clf = BertClassifier(config.bert_path, half=True, max_length=config.max_length)
        self.name = config.bert_path.rstrip("/").split("/")[-1]
        self.device = self.clf.device.type

    def predict_proba(self, texts, input_type):
        # input_type is not needed: BERT was trained on URLs and emails alike.
        return self.clf.predict_proba(texts, batch_size=32)


class ClassicalService(ModelService):
    """A saved scikit-learn TF-IDF pipeline (raw text in, probability out)."""

    kind = "classical"

    def __init__(self, config: Config):
        import joblib

        self.pipeline = joblib.load(config.classical_path)
        self.name = config.classical_path.rstrip("/").split("/")[-1].removesuffix(".joblib")

    def predict_proba(self, texts, input_type):
        return self.pipeline.predict_proba(list(texts))[:, 1]


class HeuristicService(ModelService):
    """The original regex heuristic from the web app (Python port)."""

    kind = "heuristic"
    name = "heuristic"

    def predict_proba(self, texts, input_type):
        from src.models.heuristic import analyse

        return np.array([analyse(t, input_type)["normalised"] for t in texts])


def create_service(config: Config) -> ModelService:
    services = {"bert": BertService, "classical": ClassicalService, "heuristic": lambda c: HeuristicService()}
    if config.model_kind not in services:
        raise ValueError(f"PHISHGUARD_MODEL must be one of {sorted(services)}, got {config.model_kind!r}")
    return services[config.model_kind](config)
