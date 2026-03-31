from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}

def test_simulate_validation_error():
    response = client.post("/api/v1/simulate", json={})
    assert response.status_code == 422