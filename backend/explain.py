"""
Builds the human-readable part of an analysis.

At this step only the model's own probability is explained. Rule-based
indicators and word attribution are added in the next step.
"""

ATTRIBUTION_METHOD = "none"


def explain(service, config, text, content, input_type, headers, phish_prob):
    verdict = config.verdict(phish_prob)
    explanation = (f"The {service.kind} model estimates a {100 * phish_prob:.1f}% probability that this "
                   f"{input_type} is phishing (verdict: {verdict}).")
    return {"indicators": [], "top_tokens": [], "attribution_method": ATTRIBUTION_METHOD,
            "explanation": explanation}
