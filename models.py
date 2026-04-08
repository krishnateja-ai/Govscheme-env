"""
models.py — GovScheme-Env typed models.

Uses dataclasses (Action / Observation / State pattern from OpenEnv spec).
These are the contracts between the client and the server.

Three action types map to three tasks:
  1. identify_schemes  → Task 1 (easy)
  2. rank_schemes      → Task 2 (medium)
  3. fill_form         → Task 3 (hard)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Actions (what the AI agent sends) ─────────────────────────────────────

@dataclass
class GovSchemeAction:
    """
    One action the agent can take.

    action_type tells the server which task you are answering:
      "identify_schemes"  →  fill scheme_ids
      "rank_schemes"      →  fill ranked_schemes
      "fill_form"         →  fill form_data
    """
    action_type: str                            # required, one of the three above

    # Task 1 payload
    scheme_ids: Optional[List[str]] = None      # e.g. ["PM_KISAN", "MGNREGA"]

    # Task 2 payload
    ranked_schemes: Optional[List[Dict[str, Any]]] = None
    # each item: {"scheme_id": str, "rank": int, "reason": str, "benefit_inr": int}

    # Task 3 payload
    form_data: Optional[Dict[str, Any]] = None  # {"field_name": value, ...}

    # Optional free-text (used for partial credit on reasoning)
    reasoning: Optional[str] = None


# ── Observations (what the agent receives back) ───────────────────────────

@dataclass
class GovSchemeObservation:
    """
    Everything the agent sees at each step.
    The citizen_profile contains all personal data the agent may use.
    """
    # Core task context
    citizen_profile: Dict[str, Any]             # demographics, income, land, docs
    task_name: str                              # current task id
    task_description: str                       # human-readable instruction
    available_schemes: List[Dict[str, Any]]     # 18 scheme summaries

    # Progressive context (populated in later tasks)
    identified_schemes: Optional[List[str]] = None   # given in tasks 2 & 3
    target_scheme_id: Optional[str] = None           # given in task 3
    form_template: Optional[Dict[str, Any]] = None   # field specs, task 3 only

    # Episode bookkeeping
    step_number: int = 1
    max_steps: int = 3
    cumulative_reward: float = 0.0

    # Reward from last step (0.0 on first step)
    reward: float = 0.0
    done: bool = False


# ── State (internal metadata the server exposes for debugging) ────────────

@dataclass
class GovSchemeState:
    """
    Internal episode state. Returned by GET /state.
    Useful for logging and debugging — not shown to the agent during training.
    """
    episode_id: str = ""
    step_count: int = 0
    citizen_id: Optional[str] = None
    citizen_name: Optional[str] = None
    task_name: str = ""
    done: bool = False
    cumulative_reward: float = 0.0
    best_score: float = 0.0
    rewards_history: List[float] = field(default_factory=list)
    gold_scheme_ids: List[str] = field(default_factory=list)
    top_scheme: Optional[str] = None
