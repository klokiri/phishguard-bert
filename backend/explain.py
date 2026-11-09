"""
Builds the human-readable part of an analysis from things that were actually
measured for THIS input:
  - the model's phishing probability,
  - rule-based indicators that fired (backend/indicators.py),
  - the words that most changed the model's score (backend/attribution.py).
Nothing is mentioned that did not fire or was not measured.
"""

from backend import attribution, indicators

ATTRIBUTION_METHOD = attribution.ATTRIBUTION_METHOD
_STATS = None


def _stats():
    global _STATS
    if _STATS is None:
        _STATS = indicators.load_stats()
    return _STATS


def _quote_list(items):
    return ", ".join(f"'{i}'" for i in items)


def format_prob(p: float) -> str:
    """'>99.9%' rather than a rounded '100.0%', which would overstate certainty."""
    if p > 0.999:
        return ">99.9%"
    if p < 0.001:
        return "<0.1%"
    return f"{100 * p:.1f}%"


def explain(service, config, text, content, input_type, headers, phish_prob):
    inds = indicators.build_indicators(text, input_type, _stats())
    tokens = attribution.occlusion(service, text, input_type, phish_prob,
                                   config.attribution_max_words, config.attribution_top_k)
    verdict = config.verdict(phish_prob)

    parts = [f"The {service.info()['name']} model ({service.kind}) estimates a {format_prob(phish_prob)} "
             f"probability that this {input_type} is phishing, so it is shown as {verdict}."]
    risks = [i for i in inds if i["direction"] == "risk"]
    safes = [i for i in inds if i["direction"] == "safe"]
    if risks:
        parts.append("Rule-based checks found these warning signs: "
                     + "; ".join(f"{i['name']} ({i['evidence']})" for i in risks[:3]) + ".")
    else:
        parts.append("None of the rule-based warning-sign checks fired.")
    if safes:
        parts.append("Signs more common in legitimate messages: " + "; ".join(i["name"] for i in safes[:2]) + ".")
    up = [t["token"] for t in tokens if t["direction"] == "phishing"]
    down = [t["token"] for t in tokens if t["direction"] == "legitimate"]
    if up:
        parts.append(f"Removing these words lowered the model's phishing score the most: {_quote_list(up[:3])}.")
    if down:
        parts.append(f"Removing these words raised it: {_quote_list(down[:2])}.")
    parts.append("The rule-based checks are separate from the model; the word list comes from the model itself.")

    return {"indicators": inds, "top_tokens": tokens, "attribution_method": ATTRIBUTION_METHOD,
            "explanation": " ".join(parts)}
