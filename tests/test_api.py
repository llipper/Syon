from fastapi.testclient import TestClient

from syon.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_security_analysis_without_model():
    response = client.post(
        "/v1/security-analysis",
        json={
            "model": "syon-7b",
            "code": "import pickle; pickle.loads(data)",
            "language": "python",
            "include_llm_analysis": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == "high"
    assert len(body["findings"]) >= 1


def test_security_analysis_unsupported_language():
    response = client.post(
        "/v1/security-analysis",
        json={"code": "print('hi')", "language": "brainfuck", "include_llm_analysis": False},
    )
    assert response.status_code == 400