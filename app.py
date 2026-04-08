"""
server/app.py

FastAPI server for GovScheme-Env.
Exposes all OpenEnv-required HTTP endpoints.

Endpoints:
  POST /reset   — start a new episode
  POST /step    — take one action
  GET  /state   — see internal episode state
  GET  /tasks   — list all 3 tasks
  GET  /health  — liveness check (used by HF Spaces and validator)
  GET  /schemes — browse all 18 schemes
"""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, Dict, List, Optional

# Make sure server directory is on path so imports work inside Docker
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from govscheme_environment import GovSchemeEnvironment
from models import GovSchemeAction


# ── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="GovScheme-Env",
    description=(
        "OpenEnv environment: Government Scheme Eligibility Matching. "
        "An AI agent helps Indian citizens find, rank, and apply for welfare schemes."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful between reset/step calls)
_env = GovSchemeEnvironment()


# ── Request schemas ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "scheme_identification"
    citizen_id: Optional[str] = None
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action_type: str
    scheme_ids: Optional[List[str]] = None
    ranked_schemes: Optional[List[Dict[str, Any]]] = None
    form_data: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None


# ── Helper ─────────────────────────────────────────────────────────────────

def _obs_to_dict(obs) -> Dict:
    """Convert a GovSchemeObservation dataclass to a plain dict for JSON response."""
    return dataclasses.asdict(obs)

def _state_to_dict(state) -> Dict:
    return dataclasses.asdict(state)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check. Validator pings this first."""
    return {"status": "ok", "env": "govscheme-env", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """
    Start a new episode.

    Body (all optional):
      task_name   — "scheme_identification" | "scheme_ranking" | "form_filling"
      citizen_id  — pin a citizen for reproducible eval (e.g. "CIT_001")
      seed        — integer RNG seed
    """
    valid = ["scheme_identification", "scheme_ranking", "form_filling"]
    if req.task_name not in valid:
        raise HTTPException(400, f"Invalid task_name. Choose from: {valid}")

    obs = _env.reset(
        task_name=req.task_name,
        citizen_id=req.citizen_id,
        seed=req.seed,
    )
    return JSONResponse({
        "observation": _obs_to_dict(obs),
        "reward": 0.0,
        "done": False,
        "info": {"citizen_id": _env.state.citizen_id},
    })


@app.post("/step")
def step(req: StepRequest):
    """
    Take one action in the environment.

    Body:
      action_type     — "identify_schemes" | "rank_schemes" | "fill_form"
      scheme_ids      — [list of scheme IDs]              (Task 1)
      ranked_schemes  — [{scheme_id, rank, reason, ...}]  (Task 2)
      form_data       — {field: value, ...}               (Task 3)
      reasoning       — optional free text

    Returns:
      observation, reward (0.0–1.0), done, info (score breakdown)
    """
    action = GovSchemeAction(
        action_type=req.action_type,
        scheme_ids=req.scheme_ids,
        ranked_schemes=req.ranked_schemes,
        form_data=req.form_data,
        reasoning=req.reasoning,
    )
    obs, reward, done, info = _env.step(action)
    return JSONResponse({
        "observation": _obs_to_dict(obs),
        "reward": reward,
        "done": done,
        "info": info,
    })


@app.get("/state")
def state():
    """Internal episode state (for debugging and logging)."""
    return JSONResponse(_state_to_dict(_env.state))


@app.get("/tasks")
def tasks():
    """List all 3 tasks with difficulty, description, and grader info."""
    return {"tasks": GovSchemeEnvironment.list_tasks()}


@app.get("/schemes")
def schemes():
    """Browse all 18 government schemes in the environment."""
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).parent / "schemes.json").read_text())
    return {
        "count": len(data),
        "schemes": [
            {
                "scheme_id": s["scheme_id"],
                "name": s["name"],
                "benefit_type": s["benefit_type"],
                "annual_benefit_inr": s["annual_benefit_inr"],
                "benefit_description": s["benefit_description"],
            }
            for s in data
        ],
    }


# ── Run directly for local dev ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
