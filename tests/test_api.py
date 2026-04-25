"""HTTP-level smoke tests for the FastAPI app."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_meta_lists_tasks():
    r = client.get("/meta")
    assert r.status_code == 200
    assert len(r.json()["tasks"]) >= 3


def test_reset_and_step():
    r = client.post("/reset", json={})
    assert r.status_code == 200
    assert "observation" in r.json()

    r = client.post("/step", json={"side": "hold", "quantity": 0})
    assert r.status_code == 200
    assert -1.0 <= r.json()["reward"] <= 1.0


def test_state_roundtrip():
    client.post("/reset", json={"task_id": "task_easy"})
    s = client.get("/state").json()
    r = client.post("/state", json=s)
    assert r.status_code == 200
