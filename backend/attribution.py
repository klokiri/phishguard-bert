"""
Word attribution by occlusion ("leave one word out").

Question it answers: "Which words changed the model's phishing probability
the most?" Method:
  1. p = the model's phishing probability for the full text.
  2. For each of the first N words, delete just that word and ask the model
     again: p_without.
  3. effect = logit(p) - logit(p_without), where logit(p) = log(p / (1 - p))
     is the log-odds. Positive: the word pushed the score TOWARDS phishing
     (removing it lowers the score). Negative: it pushed towards legitimate.
     Log-odds rather than plain probability because a confident model sits at
     p = 0.9999, where deleting a word moves p by only ~0.0005 even when it
     matters; in log-odds 0.9999 -> 0.999 is a clear change of 2.3.
  4. A word that appears several times keeps its largest effect. The top-k
     words by absolute effect are returned.

Honest limits (shown to users via the method name and in the report):
  - It measures one word at a time, so words that only matter together are
    missed.
  - Only the first N words are tested (config.attribution_max_words), to
    keep the response fast; BERT itself reads up to 512 tokens.
  - It describes what the model reacts to, not whether that is a good reason.
It is model-agnostic: it only calls ModelService.predict_proba, so it works
for BERT, the classical models and the heuristic alike.
"""

import math
import re

ATTRIBUTION_METHOD = "occlusion (leave-one-word-out), effect in log-odds"
_EPS = 1e-6  # keeps log-odds finite when a model outputs exactly 0 or 1


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))
# A "word" is a run of letters/digits, so URLs split into parts:
# "paypa1-secure.tk/login" -> paypa1, secure, tk, login.
WORD = re.compile(r"[A-Za-z0-9]+")


def occlusion(service, text: str, input_type: str, base_prob: float, max_words: int, top_k: int) -> list[dict]:
    spans = [m.span() for m in WORD.finditer(text)][:max_words]
    if not spans:
        return []
    variants = [text[:a] + text[b:] for a, b in spans]
    probs = service.predict_proba(variants, input_type)  # one batched call

    base = logit(base_prob)
    best = {}  # lower-cased word -> effect with the largest absolute value
    for (a, b), p_without in zip(spans, probs):
        word = text[a:b].lower()
        effect = base - logit(float(p_without))
        if word not in best or abs(effect) > abs(best[word]):
            best[word] = effect

    ranked = sorted(best.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    # Words that change the log-odds by less than 0.05 (about a 1% change in
    # the odds) are dropped as "no measurable effect".
    return [{"token": w, "effect": round(e, 3), "direction": "phishing" if e > 0 else "legitimate"}
            for w, e in ranked if abs(e) >= 0.05]
