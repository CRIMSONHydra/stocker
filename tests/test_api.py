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


def test_ohlcv_returns_bars():
    r = client.get("/ohlcv", params={"task_id": "task_easy"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert len(body["bars"]) > 0
    bar = body["bars"][0]
    assert {"time", "open", "high", "low", "close", "volume", "in_episode"} <= bar.keys()
    assert bar["high"] >= bar["low"]


def test_ohlcv_unknown_task_404():
    r = client.get("/ohlcv", params={"task_id": "nope"})
    assert r.status_code == 404


def test_council_after_reset():
    client.post("/reset", json={"task_id": "task_easy"})
    r = client.get("/council")
    assert r.status_code == 200
    body = r.json()
    assert len(body["votes"]) == 7
    assert body["action"]["side"] in {"buy", "sell", "hold"}


def test_training_metrics_returns_summary():
    r = client.get("/training/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"completed", "no_runs"}
    if body["status"] == "completed":
        assert len(body["summary"]) >= 1
        assert "alpha_pct" in body["summary"][0]
