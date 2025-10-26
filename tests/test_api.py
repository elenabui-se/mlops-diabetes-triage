from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_ok():
    payload = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
        "s5": 0.02, "s6": -0.001
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()

def test_bad_input_returns_json_error():
    r = client.post("/predict", json={"age": 0.1})  # missing fields
    assert r.status_code in (400, 422)
