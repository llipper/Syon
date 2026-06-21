from config.constants import BENCHMARK_NAMES, MODEL_SIZES, SUPPORTED_LANGUAGES
from config.settings import Settings


def test_constants():
    assert len(SUPPORTED_LANGUAGES) >= 15
    assert "syon-7b" in MODEL_SIZES
    assert "humaneval" in BENCHMARK_NAMES


def test_settings_defaults():
    settings = Settings()
    assert settings.api.api_port == 8000
    assert settings.training.model_name in MODEL_SIZES