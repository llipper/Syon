from fastapi.testclient import TestClient

from api.rest.app import app

client = TestClient(app)


def test_health_v2():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    models = body.get("data") or body.get("models", [])
    assert any(m["id"] == "syon-7b" for m in models)


def test_security_analysis_v2():
    response = client.post(
        "/v1/security-analysis",
        json={
            "code": "import pickle; pickle.loads(data)",
            "language": "python",
            "include_llm_analysis": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["risk_level"] == "high"