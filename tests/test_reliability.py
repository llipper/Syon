"""Testes de confiabilidade — reprodutibilidade e consistência."""

from syon.config import SyonConfig
from syon.security.analyzer import SecurityAnalyzer


def test_config_reproducibility():
    a = SyonConfig.load("syon-7b")
    b = SyonConfig.load("syon-7b")
    assert a.context_window == b.context_window
    assert a.model_name == b.model_name


def test_analysis_consistency():
    code = "os.system(cmd)"
    r1 = SecurityAnalyzer().analyze(code)
    r2 = SecurityAnalyzer().analyze(code)
    assert r1.risk_level == r2.risk_level
    assert len(r1.findings) == len(r2.findings)


def test_dataset_composition_weights():
    from training.dataset.composition import load_composition

    comp = load_composition()
    assert comp.validate_weights()