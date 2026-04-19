import pytest
from unittest.mock import patch, MagicMock
from shared.model.inference import load_model, run


def test_run_extracts_model_response():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = (
        "<start_of_turn>user\nQ<end_of_turn>\n"
        "<start_of_turn>model\nThe answer is 42.<end_of_turn>"
    )
    mock_model.generate.return_value = [MagicMock()]
    mock_model.device = "cpu"

    with patch.object(mock_tokenizer, "__call__", return_value={"input_ids": MagicMock()}):
        result = run(mock_model, mock_tokenizer, "Q", "")

    assert "42" in result


def test_run_with_context():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = (
        "<start_of_turn>model\nValves V-101 and V-102.<end_of_turn>"
    )
    mock_model.generate.return_value = [MagicMock()]
    mock_model.device = "cpu"

    with patch.object(mock_tokenizer, "__call__", return_value={"input_ids": MagicMock()}):
        result = run(mock_model, mock_tokenizer, "Isolate P-101", '{"component": "P-101"}')

    assert "V-101" in result


def test_load_model_raises_on_empty_adapter_dir():
    with pytest.raises(ValueError, match="adapter_dir"):
        load_model("")


def test_run_fallback_when_no_model_turn():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "Some response without turn markers"
    mock_model.generate.return_value = [MagicMock()]
    mock_model.device = "cpu"

    with patch.object(mock_tokenizer, "__call__", return_value={"input_ids": MagicMock()}):
        result = run(mock_model, mock_tokenizer, "Q", "")

    assert result == "Some response without turn markers"
