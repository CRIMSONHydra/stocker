"""HTTP client for the Stocker OpenEnv environment."""

import requests


class StockerClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "task_easy") -> dict:
        r = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, side: str, quantity: int = 0) -> dict:
        r = requests.post(
            f"{self.base_url}/step",
            json={"side": side, "quantity": quantity},
        )
        r.raise_for_status()
        return r.json()

    def get_state(self) -> dict:
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def restore_state(self, state: dict) -> dict:
        r = requests.post(f"{self.base_url}/state", json=state)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()
