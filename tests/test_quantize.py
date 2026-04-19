# tests/test_quantize.py
import pytest
from shared.model.quantize import export_gguf, SUPPORTED_QUANT_TYPES


def test_export_gguf_raises_on_empty_adapter_dir():
    with pytest.raises(ValueError, match="adapter_dir"):
        export_gguf("", "/some/output.gguf")


def test_export_gguf_raises_on_empty_output_path():
    with pytest.raises(ValueError, match="output_path"):
        export_gguf("/some/adapter", "")


def test_export_gguf_raises_on_invalid_quant_type():
    with pytest.raises(ValueError, match="quant_type"):
        export_gguf("/some/adapter", "/some/output.gguf", quant_type="invalid")


def test_export_gguf_raises_on_missing_adapter_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_gguf(str(tmp_path / "nonexistent"), str(tmp_path / "out.gguf"))


def test_supported_quant_types_includes_common_formats():
    assert "q4_k_m" in SUPPORTED_QUANT_TYPES
    assert "q8_0" in SUPPORTED_QUANT_TYPES
