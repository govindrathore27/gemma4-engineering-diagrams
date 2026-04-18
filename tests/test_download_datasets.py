import pytest
from unittest.mock import patch, MagicMock
from shared.data_pipeline.download_datasets import DATASETS, download_all

def test_datasets_registry_has_all_five():
    assert set(DATASETS.keys()) == {
        "pid2graph", "pid_symbols", "cghd", "as1100", "dimensioning"
    }

def test_download_all_calls_correct_handler():
    with patch("shared.data_pipeline.download_datasets.download_zenodo") as mock_z, \
         patch("shared.data_pipeline.download_datasets.download_kaggle_dataset") as mock_k, \
         patch("shared.data_pipeline.download_datasets.download_huggingface") as mock_hf:
        download_all(names=["pid2graph"])
        mock_z.assert_called_once()
        mock_k.assert_not_called()
