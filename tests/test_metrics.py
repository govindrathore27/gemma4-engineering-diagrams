import json
import pytest
from shared.eval.metrics import token_f1, exact_match, bleu_1, evaluate_batch


def test_token_f1_perfect():
    assert token_f1("valve A valve B", "valve A valve B") == 1.0


def test_token_f1_partial():
    score = token_f1("valve A", "valve A valve B")
    assert 0.0 < score < 1.0


def test_token_f1_no_overlap():
    assert token_f1("pump X", "valve Y") == 0.0


def test_exact_match_json_lists():
    pred = json.dumps({"components": ["V-101", "V-102"]})
    exp = json.dumps({"components": ["V-102", "V-101"]})
    assert exact_match(pred, exp) == 1.0


def test_bleu_1_full_overlap():
    assert bleu_1("the quick brown fox", "the quick brown fox") == 1.0


def test_evaluate_batch_returns_mean():
    result = evaluate_batch(["a b", "c d"], ["a b", "c d"], metric="f1")
    assert result["mean"] == pytest.approx(1.0)
    assert result["metric"] == "f1"


def test_evaluate_batch_empty_lists():
    result = evaluate_batch([], [], metric="f1")
    assert result["mean"] == 0.0
    assert result["scores"] == []


def test_evaluate_batch_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        evaluate_batch(["a"], ["a", "b"], metric="f1")


def test_token_f1_both_empty():
    assert token_f1("", "") == 1.0


def test_bleu_1_repeated_token_not_inflated():
    score = bleu_1("valve valve valve valve", "valve pump")
    assert score < 1.0


def test_exact_match_isolation_valves_json():
    pred = json.dumps({"pump": "P-101", "isolation_valves": ["V-101", "V-102"]})
    exp = json.dumps({"pump": "P-101", "isolation_valves": ["V-101", "V-102"]})
    assert exact_match(pred, exp) == 1.0


def test_exact_match_plain_string():
    assert exact_match("valve A", "valve A") == 1.0
    assert exact_match("valve A", "valve B") == 0.0
