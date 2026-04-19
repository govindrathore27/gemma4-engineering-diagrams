import json
from collections import Counter


def token_f1(predicted: str, expected: str) -> float:
    """
    Calculate F1 score based on token overlap using Counter intersection.

    Args:
        predicted: Predicted text string
        expected: Expected (ground truth) text string

    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = Counter(predicted.lower().split())
    exp_tokens = Counter(expected.lower().split())
    if not pred_tokens and not exp_tokens:
        return 1.0
    common = sum((pred_tokens & exp_tokens).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_tokens.values())
    recall = common / sum(exp_tokens.values())
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str, expected: str) -> float:
    """
    Check if predicted and expected match exactly.

    For JSON with "components" field, compares as sets (order-independent).
    For plain strings, does direct string comparison.

    Args:
        predicted: Predicted text (may be JSON)
        expected: Expected text (may be JSON)

    Returns:
        1.0 if match, 0.0 otherwise
    """
    try:
        pred_obj = json.loads(predicted)
        exp_obj = json.loads(expected)
        if "components" in pred_obj and "components" in exp_obj:
            return float(set(pred_obj["components"]) == set(exp_obj["components"]))
        return float(pred_obj == exp_obj)
    except (json.JSONDecodeError, AttributeError):
        return float(predicted.strip() == expected.strip())


def bleu_1(predicted: str, expected: str) -> float:
    """
    Calculate BLEU-1 score (unigram precision).

    Measures the fraction of predicted tokens that appear in expected.

    Args:
        predicted: Predicted text string
        expected: Expected text string

    Returns:
        Score between 0.0 and 1.0
    """
    pred_tokens = predicted.lower().split()
    exp_counts = Counter(expected.lower().split())
    if not pred_tokens:
        return 0.0
    clipped = sum(min(pred_tokens.count(t), exp_counts[t]) for t in set(pred_tokens))
    return clipped / len(pred_tokens)


def evaluate_batch(
    predictions: list[str], expected: list[str], metric: str = "f1"
) -> dict:
    """
    Evaluate a batch of predictions against expected values.

    Args:
        predictions: List of predicted strings
        expected: List of expected strings
        metric: Metric to use ("f1", "exact", or "bleu")

    Returns:
        Dictionary with keys:
            - metric: The metric name used
            - mean: Mean score across all pairs
            - scores: List of individual scores

    Raises:
        ValueError: If metric is not recognized or lists have unequal length
    """
    if len(predictions) != len(expected):
        raise ValueError(
            f"predictions and expected must have equal length, "
            f"got {len(predictions)} and {len(expected)}"
        )
    fns = {"f1": token_f1, "exact": exact_match, "bleu": bleu_1}
    if metric not in fns:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(fns)}")
    if not predictions:
        return {"metric": metric, "mean": 0.0, "scores": []}
    scores = [fns[metric](p, e) for p, e in zip(predictions, expected)]
    return {"metric": metric, "mean": sum(scores) / len(scores), "scores": scores}
