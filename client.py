"""
client.py — GovSchemeEnv client.

Your teammate imports this in inference.py:
    from govscheme_env import GovSchemeEnv, GovSchemeAction

It wraps all HTTP calls so inference.py never touches requests directly.
Usage:
    client = GovSchemeEnv(base_url="http://localhost:7860")
    obs_data = client.reset(task_name="scheme_identification", citizen_id="CIT_001", seed=42)
    result   = client.step(GovSchemeAction(action_type="identify_schemes", scheme_ids=[...]))
    state    = client.state()
    client.close()
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests

from models import GovSchemeAction, GovSchemeObservation, GovSchemeState


class GovSchemeEnv:
    """
    HTTP client for GovScheme-Env server.

    Args:
        base_url: URL where the server is running.
                  Local:      "http://localhost:7860"
                  HF Space:   "https://your-username-govscheme-env.hf.space"
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ── Core OpenEnv methods ──────────────────────────────────────────────

    def reset(
        self,
        task_name: str = "scheme_identification",
        citizen_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Start a new episode.

        Returns the raw response dict:
          {observation: {...}, reward: 0.0, done: false, info: {...}}
        """
        payload = {"task_name": task_name}
        if citizen_id:
            payload["citizen_id"] = citizen_id
        if seed is not None:
            payload["seed"] = seed

        resp = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: GovSchemeAction) -> Dict[str, Any]:
        """
        Take one action. Returns the raw response dict:
          {observation: {...}, reward: float, done: bool, info: {...}}
        """
        payload = {"action_type": action.action_type}
        if action.scheme_ids is not None:
            payload["scheme_ids"] = action.scheme_ids
        if action.ranked_schemes is not None:
            payload["ranked_schemes"] = action.ranked_schemes
        if action.form_data is not None:
            payload["form_data"] = action.form_data
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning

        resp = self._session.post(f"{self.base_url}/step", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Return current internal episode state."""
        resp = self._session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Clean up the HTTP session."""
        self._session.close()

    def health(self) -> bool:
        """Returns True if server is up and responding."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Context manager support ───────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
