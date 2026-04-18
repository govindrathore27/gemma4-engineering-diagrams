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
        pred_set = set(pred_obj.get("components", [str(pred_obj)]))
        exp_set = set(exp_obj.get("components", [str(exp_obj)]))
        return float(pred_set == exp_set)
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
    pred = predicted.lower().split()
    exp_set = set(expected.lower().split())
    if not pred:
        return 0.0
    return sum(1 for t in pred if t in exp_set) / len(pred)


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
        ValueError: If metric is not recognized
    """
    fns = {"f1": token_f1, "exact": exact_match, "bleu": bleu_1}
    if metric not in fns:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(fns)}")
    scores = [fns[metric](p, e) for p, e in zip(predictions, expected)]
    return {"metric": metric, "mean": sum(scores) / len(scores), "scores": scores}
