from syon.security.analyzer import SecurityAnalyzer


def test_detects_pickle_deserialization():
    code = "import pickle; data = pickle.loads(user_input)"
    result = SecurityAnalyzer().analyze(code, language="python")
    assert result.risk_level == "high"
    assert any(f.cwe == "CWE-502" for f in result.findings)


def test_clean_code_low_risk():
    code = "def add(a, b):\n    return a + b"
    result = SecurityAnalyzer().analyze(code, language="python")
    assert result.risk_level == "low"
    assert result.cvss_estimate == 0.0


def test_owasp_mapping():
    code = "cursor.execute('SELECT * FROM users WHERE id=' + user_id)"
    result = SecurityAnalyzer().analyze(code, language="python")
    assert "A03:2021-Injection" in result.owasp_categories