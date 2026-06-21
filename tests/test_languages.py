from syon.code.languages import is_supported, normalize_language


def test_normalize_aliases():
    assert normalize_language("js") == "javascript"
    assert normalize_language("C#") == "csharp"


def test_supported_languages_from_readme():
    for lang in ["python", "rust", "solidity", "bash"]:
        assert is_supported(lang)