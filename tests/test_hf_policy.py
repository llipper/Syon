import pytest

from training.hf.policy import assert_dataset_allowed, assert_no_external_model


def test_blocks_external_models():
    with pytest.raises(ValueError, match="bloqueado"):
        assert_no_external_model("google/gemma-2b")


def test_allows_syon_hub():
    assert_no_external_model("regyfelipe/syon-3")


def test_dataset_allowlist():
    assert_dataset_allowed("nicholasKluge/instruct-aira-dataset-v3")


def test_dataset_blocked():
    with pytest.raises(ValueError, match="não autorizado"):
        assert_dataset_allowed("some/random-model-weights")