"""Pure-Python QA evaluation helpers using SQuAD-style text normalization.

These are diagnostic metrics, not evidence that this repository reaches any
particular SQuAD benchmark score.
"""

from collections import Counter
import re
import string


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles and normalize whitespace."""
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> float:
    """Return 1 if prediction exactly matches any normalized reference."""
    if not references:
        raise ValueError("At least one reference answer is required.")
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(ref) for ref in references))


def token_f1(prediction: str, references: list[str]) -> float:
    """Best token-overlap F1 across alternative acceptable answers."""
    if not references:
        raise ValueError("At least one reference answer is required.")

    pred_tokens = normalize_answer(prediction).split()

    def score(reference: str) -> float:
        ref_tokens = normalize_answer(reference).split()
        if not pred_tokens or not ref_tokens:
            return float(pred_tokens == ref_tokens)
        common = Counter(pred_tokens) & Counter(ref_tokens)
        shared = sum(common.values())
        if shared == 0:
            return 0.0
        precision = shared / len(pred_tokens)
        recall = shared / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(score(ref) for ref in references)
