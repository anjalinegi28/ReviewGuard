"""
When a review gets flagged, "trust me, the model says so" isn't good enough
for a trust & safety analyst to act on. This wraps SHAP over the HF
text-classification pipeline so a flag comes with the actual words that
pushed the prediction toward the sentiment that clashed with the rating.

Falls back to a lightweight lexicon-overlap explainer when SHAP or a real
pipeline isn't available (dev/tests without a trained checkpoint) so the API
response shape stays consistent either way.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TokenContribution:
    token: str
    weight: float  # signed: positive pushes toward the predicted (mismatching) sentiment


@dataclass
class Explanation:
    predicted_label: str
    top_tokens: List[TokenContribution]
    method: str  # "shap" or "lexicon_fallback"


class ShapExplainer:
    """
    pipeline: a transformers.pipeline("text-classification", ...) instance,
    same one used to back HFPipelineClassifier for consistency between the
    score and the explanation.
    """

    def __init__(self, pipeline, top_k: int = 8):
        import shap
        self._shap = shap
        self.pipeline = pipeline
        self.top_k = top_k
        self._explainer = shap.Explainer(pipeline)

    def explain(self, text: str, predicted_label: str) -> Explanation:
        shap_values = self._explainer([text])
        # shap_values[0] shape: (tokens, classes) for text classification pipelines
        tokens = shap_values.data[0]
        label_names = list(shap_values.output_names)
        try:
            class_idx = label_names.index(predicted_label)
        except ValueError:
            class_idx = 0

        contributions = [
            TokenContribution(token=str(tok), weight=float(shap_values.values[0][i][class_idx]))
            for i, tok in enumerate(tokens)
        ]
        contributions.sort(key=lambda c: abs(c.weight), reverse=True)

        return Explanation(
            predicted_label=predicted_label,
            top_tokens=contributions[: self.top_k],
            method="shap",
        )


class LexiconFallbackExplainer:
    """No SHAP, no trained model required - just flags which lexicon words fired."""

    def __init__(self, top_k: int = 8):
        from src.mismatch.classifiers import _POSITIVE_WORDS, _NEGATIVE_WORDS
        self._pos = _POSITIVE_WORDS
        self._neg = _NEGATIVE_WORDS
        self.top_k = top_k

    def explain(self, text: str, predicted_label: str) -> Explanation:
        import re
        words = re.findall(r"[a-zA-Z']+", text.lower())
        contributions = []
        for w in words:
            if w in self._pos:
                contributions.append(TokenContribution(token=w, weight=1.0))
            elif w in self._neg:
                contributions.append(TokenContribution(token=w, weight=-1.0))

        if predicted_label == "negative":
            contributions = [TokenContribution(c.token, -abs(c.weight)) for c in contributions]
        elif predicted_label == "positive":
            contributions = [TokenContribution(c.token, abs(c.weight)) for c in contributions]

        return Explanation(
            predicted_label=predicted_label,
            top_tokens=contributions[: self.top_k],
            method="lexicon_fallback",
        )
