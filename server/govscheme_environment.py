"""
govscheme_environment.py

The GovSchemeEnvironment class.
Implements reset() / step() / state  — the three OpenEnv core methods.

How an episode works:
  1. Client calls reset(task_name, citizen_id, seed)
     -> environment picks a citizen, pre-computes gold answers, returns observation
  2. Client calls step(action) up to 3 times
     -> grader scores the action, returns (observation, reward, done)
  3. done=True when max steps hit or perfect score achieved
"""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import our graders and eligibility engine (same directory)
from eligibility import get_eligible_schemes, rank_schemes_by_benefit
from graders import (
    grade_scheme_identification,
    grade_scheme_ranking,
    grade_form_filling,
)

# ── Task configuration ────────────────────────────────────────────────────

MAX_STEPS = 3   # agent gets 3 attempts per task; best score is kept

TASK_INSTRUCTIONS = {
    "scheme_identification": (
        "You are a government welfare officer. Read the citizen profile carefully. "
        "Identify ALL central government schemes this citizen is eligible for. "
        "Check age, income, caste, occupation, land ownership, gender, and document availability. "
        "Return action_type='identify_schemes' with scheme_ids=[list of scheme IDs]. "
        "Use only IDs from available_schemes. Being inclusive is better than missing schemes."
    ),
    "scheme_ranking": (
        "You are a welfare advisor. The citizen's eligible schemes are already identified. "
        "Rank ALL eligible schemes from most to least beneficial by annual INR value. "
        "Schemes with benefit_inr=0 (credit/loan access) go at the bottom. "
        "Return action_type='rank_schemes' with ranked_schemes=[{scheme_id, rank, reason, benefit_inr}]. "
        "Your 'reason' must mention the actual INR benefit amount and why it suits this citizen."
    ),
    "form_filling": (
        "You are a government application assistant. Fill the complete application form "
        "for the scheme indicated in target_scheme_id. Use ONLY data from the citizen profile. "
        "Do NOT invent or guess any value. "
        "Critical formats: Aadhaar=12 digits starting 2-9, IFSC=4letters+0+6alphanumeric, "
        "Mobile=10 digits starting 6-9, Date=DD/MM/YYYY. "
        "Return action_type='fill_form' with form_data={field_name: value}."
    ),
}

# ── Environment class ─────────────────────────────────────────────────────

class GovSchemeEnvironment:
    """
    Government Scheme Eligibility Matching Environment.

    Attributes loaded at startup (from JSON files in same directory):
        _schemes   -- 18 Indian government schemes with eligibility rules
        _citizens  -- 10 diverse citizen profiles with verified gold labels
    """

    def __init__(self):
        base = Path(__file__).parent
        self._schemes: List[Dict] = json.loads((base / "schemes.json").read_text())
        self._citizens: List[Dict] = json.loads((base / "citizens.json").read_text())
        self._scheme_map: Dict[str, Dict] = {s["scheme_id"]: s for s in self._schemes}

        # Episode state (initialised properly in reset())
        self._citizen: Optional[Dict] = None
        self._task_name: str = "scheme_identification"
        self._episode_id: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._best_score: float = 0.0
        self._rewards_history: List[float] = []
        self._gold_scheme_ids: List[str] = []
        self._gold_ranked: List[Dict] = []
        self._rng = random.Random()

    # ── OpenEnv required methods ──────────────────────────────────────────

    def reset(
        self,
        task_name: str = "scheme_identification",
        citizen_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> "GovSchemeObservation":
        """
        Start a new episode.

        Args:
            task_name:  Which task to run. One of:
                        'scheme_identification' | 'scheme_ranking' | 'form_filling'
            citizen_id: Pin a specific citizen for reproducible evaluation.
                        Leave None for random selection.
            seed:       RNG seed (set this for reproducible runs).

        Returns:
            GovSchemeObservation -- the first observation the agent sees.
        """
        from models import GovSchemeObservation  # local import to avoid circular

        if seed is not None:
            self._rng = random.Random(seed)

        # Pick citizen
        if citizen_id:
            matches = [c for c in self._citizens if c["citizen_id"] == citizen_id]
            self._citizen = matches[0] if matches else self._rng.choice(self._citizens)
        else:
            self._citizen = self._rng.choice(self._citizens)

        # Reset episode state
        self._task_name = task_name
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._best_score = 0.0
        self._rewards_history = []

        # Pre-compute gold answers (never shown to agent; only used by graders)
        self._gold_scheme_ids = get_eligible_schemes(self._citizen, self._schemes)
        self._gold_ranked = rank_schemes_by_benefit(self._gold_scheme_ids, self._schemes)

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: "GovSchemeAction") -> Tuple["GovSchemeObservation", float, bool, Dict]:
        """
        Process one agent action.

        Returns:
            observation  -- updated observation
            reward       -- score for this step, float in [0.0, 1.0]
            done         -- True when episode is over
            info         -- dict with score breakdown (for logging)
        """
        from models import GovSchemeObservation

        if self._done:
            obs = self._build_observation(reward=0.0, done=True)
            return obs, 0.0, True, {"error": "Episode already done. Call reset()."}

        self._step_count += 1

        # Grade the action
        reward, info = self._grade(action)

        # Track reward history
        self._best_score = max(self._best_score, reward)
        self._cumulative_reward += reward
        self._rewards_history.append(round(reward, 4))

        # Episode ends at max steps or perfect score
        done = (self._step_count >= MAX_STEPS) or (reward >= 0.99)
        self._done = done

        obs = self._build_observation(reward=reward, done=done)
        info.update({
            "step": self._step_count,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "best_score": round(self._best_score, 4),
        })
        return obs, round(reward, 4), done, info

    @property
    def state(self) -> "GovSchemeState":
        """
        Internal episode metadata (for debugging and logging).
        This is what GET /state returns.
        """
        from models import GovSchemeState
        return GovSchemeState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            citizen_id=self._citizen["citizen_id"] if self._citizen else None,
            citizen_name=self._citizen.get("name") if self._citizen else None,
            task_name=self._task_name,
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            best_score=round(self._best_score, 4),
            rewards_history=self._rewards_history,
            gold_scheme_ids=self._gold_scheme_ids,
            top_scheme=self._gold_ranked[0]["scheme_id"] if self._gold_ranked else None,
        )

    # ── Helper methods ────────────────────────────────────────────────────

    def _build_observation(self, reward: float, done: bool) -> "GovSchemeObservation":
        """Build the observation dict the agent sees. Gold labels are stripped out."""
        from models import GovSchemeObservation

        # Strip internal gold labels before handing to agent
        citizen_view = {
            k: v for k, v in self._citizen.items()
            if k not in ("eligible_schemes", "top_scheme_by_benefit", "task_difficulty")
        }

        scheme_summaries = [
            {
                "scheme_id": s["scheme_id"],
                "name": s["name"],
                "ministry": s["ministry"],
                "benefit_type": s["benefit_type"],
                "annual_benefit_inr": s["annual_benefit_inr"],
                "benefit_description": s["benefit_description"],
            }
            for s in self._schemes
        ]

        # Task 3: give the agent the form template for the top scheme
        form_template = None
        target_scheme_id = None
        if self._task_name == "form_filling" and self._gold_ranked:
            target_scheme_id = self._gold_ranked[0]["scheme_id"]
            target_scheme = self._scheme_map.get(target_scheme_id, {})
            form_template = target_scheme.get("application_fields", {})

        return GovSchemeObservation(
            citizen_profile=citizen_view,
            task_name=self._task_name,
            task_description=TASK_INSTRUCTIONS[self._task_name],
            available_schemes=scheme_summaries,
            identified_schemes=self._gold_scheme_ids if self._task_name != "scheme_identification" else None,
            target_scheme_id=target_scheme_id,
            form_template=form_template,
            step_number=self._step_count,
            max_steps=MAX_STEPS,
            cumulative_reward=round(self._cumulative_reward, 4),
            reward=reward,
            done=done,
        )

    def _grade(self, action: "GovSchemeAction") -> Tuple[float, Dict]:
        """Route action to the correct grader."""
        if self._task_name == "scheme_identification":
            predicted = action.scheme_ids or []
            score, breakdown = grade_scheme_identification(predicted, self._gold_scheme_ids)
            return score, {"task": "scheme_identification", "score": score, **breakdown}

        elif self._task_name == "scheme_ranking":
            ranked = action.ranked_schemes or []
            score, breakdown = grade_scheme_ranking(ranked, self._gold_ranked, self._citizen)
            return score, {"task": "scheme_ranking", "score": score, **breakdown}

        elif self._task_name == "form_filling":
            if not self._gold_ranked:
                return 0.0, {"error": "citizen has no eligible schemes"}
            top_id = self._gold_ranked[0]["scheme_id"]
            scheme = self._scheme_map.get(top_id, {})
            form_data = action.form_data or {}
            score, breakdown = grade_form_filling(form_data, scheme, self._citizen)
            return score, {"task": "form_filling", "scheme": top_id, "score": score, **breakdown}

        return 0.0, {"error": f"Unknown task: {self._task_name}"}

    @staticmethod
    def list_tasks() -> List[Dict[str, Any]]:
        """Metadata about all 3 tasks (used by GET /tasks endpoint)."""
        return [
            {
                "task_id": "scheme_identification",
                "difficulty": "easy",
                "description": "Identify all government schemes a citizen is eligible for",
                "max_steps": MAX_STEPS,
                "score_range": [0.0, 1.0],
                "grader": "F-beta (beta=1.5), recall-weighted",
            },
            {
                "task_id": "scheme_ranking",
                "difficulty": "medium",
                "description": "Rank eligible schemes by annual benefit value with justification",
                "max_steps": MAX_STEPS,
                "score_range": [0.0, 1.0],
                "grader": "NDCG@5 (50%) + benefit accuracy (30%) + reasoning quality (20%)",
            },
            {
                "task_id": "form_filling",
                "difficulty": "hard",
                "description": "Auto-fill a government application form from citizen profile without hallucinating",
                "max_steps": MAX_STEPS,
                "score_range": [0.0, 1.0],
                "grader": "field accuracy (40%) + format validity (30%) + completeness (20%) - hallucination penalty",
            },
        ]
