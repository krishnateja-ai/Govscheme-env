"""
inference.py — GovScheme-Env Baseline Inference Script
=======================================================
MANDATORY — placed at root of project as required by hackathon rules.

Reads from environment variables:
  API_BASE_URL  — LLM API endpoint  (default: HuggingFace router)
  MODEL_NAME    — model to use      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      — your HF API key
  ENV_URL       — where the env server is running (default: localhost:7860)

Stdout format (exactly as required by hackathon evaluator):
  [START] task=<task_name> env=govscheme-env model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Run:
  python inference.py
"""

import json
import os
import re
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS         = 3       # max steps per task (matches env setting)
TEMPERATURE       = 0.2     # low = more deterministic = more reproducible
MAX_TOKENS        = 1500
SUCCESS_THRESHOLD = 0.5     # score >= 0.5 counts as success

# Fixed citizen + seed for reproducible baseline scores
TASK_CONFIG = {
    "scheme_identification": {"citizen_id": "CIT_001", "seed": 42},  # easy: rural farmer UP
    "scheme_ranking":        {"citizen_id": "CIT_006", "seed": 42},  # medium: weaver WB
    "form_filling":          {"citizen_id": "CIT_009", "seed": 42},  # hard: SC farmer Odisha
}

TASKS = list(TASK_CONFIG.keys())


# ── Mandatory stdout loggers ───────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    safe_action = action.replace("\n", " ")[:200]
    print(f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── System prompts per task ────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "scheme_identification": textwrap.dedent("""
        You are a government welfare officer helping Indian citizens access welfare schemes.
        Given a citizen profile and a list of 18 Indian government schemes, identify ALL schemes
        this citizen is eligible for.

        Check these eligibility factors carefully:
        - Age limits (age_min, age_max)
        - Income limits (annual_income_inr, annual_family_income_inr)
        - Caste (SC, ST, OBC, General)
        - Occupation (farmer, student, weaver, etc.)
        - Gender
        - Land ownership (acres)
        - Has Aadhaar, has bank account
        - Rural vs urban
        - House type (kachha qualifies for housing schemes)
        - NOT a government employee, NOT an income taxpayer

        Respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
        {
          "action_type": "identify_schemes",
          "scheme_ids": ["SCHEME_ID_1", "SCHEME_ID_2"],
          "reasoning": "Brief explanation of why each scheme was included"
        }

        Use only scheme IDs from the available_schemes list.
        When in doubt, include the scheme — missing one hurts more than including an extra.
    """).strip(),

    "scheme_ranking": textwrap.dedent("""
        You are a welfare advisor ranking government schemes by benefit value for a citizen.
        Rank ALL eligible schemes from most to least beneficial based on annual_benefit_inr.
        Schemes with benefit_inr=0 (credit/loan access schemes) go at the bottom.

        Respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
        {
          "action_type": "rank_schemes",
          "ranked_schemes": [
            {
              "scheme_id": "AYUSHMAN_BHARAT",
              "rank": 1,
              "reason": "Highest value: ₹5,00,000 health insurance. Citizen income ₹72,000 qualifies.",
              "benefit_inr": 500000
            },
            ...
          ],
          "reasoning": "Overall ranking rationale"
        }

        Your "reason" for each scheme MUST mention:
        - The actual INR benefit amount
        - Why this citizen qualifies (caste, income, occupation, etc.)
    """).strip(),

    "form_filling": textwrap.dedent("""
        You are a government application assistant filling forms for citizens.
        Fill the application form for the target scheme using ONLY data from the citizen profile.
        DO NOT invent, guess, or hallucinate any values.

        STRICT format rules — get these exactly right:
        - Aadhaar number: 12 digits, must start with 2-9 (e.g. "234567890123")
        - IFSC code: 4 uppercase letters + 0 + 6 alphanumeric (e.g. "SBIN0001234")
        - Mobile number: 10 digits, must start with 6, 7, 8, or 9 (e.g. "9876543210")
        - Date of birth: DD/MM/YYYY format (e.g. "15/03/1986")
        - category: use the caste field value (SC / ST / OBC / General)
        - gender: use exactly "Male", "Female", or "Other"

        If a field value is not in the citizen profile, leave it out entirely — do not guess.

        Respond ONLY with valid JSON, no markdown:
        {
          "action_type": "fill_form",
          "form_data": {
            "applicant_name": "...",
            "aadhaar_number": "...",
            ...
          },
          "reasoning": "Explanation of how you mapped each field"
        }
    """).strip(),
}


# ── LLM call ──────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, task: str, obs: Dict) -> Dict:
    """Call the LLM with the current observation and return parsed JSON action."""
    user_prompt = _build_prompt(task, obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps response in ```json ... ```
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", flush=True)
        return {"action_type": "identify_schemes", "scheme_ids": [], "reasoning": "parse_error"}
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {"action_type": "identify_schemes", "scheme_ids": [], "reasoning": "llm_error"}


def _build_prompt(task: str, obs: Dict) -> str:
    """Build a clear user prompt from the current observation."""
    citizen = obs.get("citizen_profile", {})
    available = obs.get("available_schemes", [])
    identified = obs.get("identified_schemes")
    form_template = obs.get("form_template")
    target_scheme = obs.get("target_scheme_id")

    lines = [
        f"TASK: {obs.get('task_description', '')}",
        "",
        "=== CITIZEN PROFILE ===",
        json.dumps(citizen, indent=2),
        "",
    ]

    if task == "scheme_identification":
        lines += [
            "=== AVAILABLE SCHEMES (use only these scheme IDs) ===",
            json.dumps(
                [{"scheme_id": s["scheme_id"], "name": s["name"],
                  "benefit_type": s["benefit_type"], "annual_benefit_inr": s["annual_benefit_inr"]}
                 for s in available],
                indent=2
            ),
        ]
    elif task == "scheme_ranking":
        eligible = [s for s in available if identified and s["scheme_id"] in identified]
        lines += [
            "=== ELIGIBLE SCHEMES TO RANK (rank all of these) ===",
            json.dumps(
                [{"scheme_id": s["scheme_id"], "name": s["name"],
                  "annual_benefit_inr": s["annual_benefit_inr"],
                  "benefit_description": s["benefit_description"]}
                 for s in eligible],
                indent=2
            ),
        ]
    elif task == "form_filling":
        lines += [
            f"=== TARGET SCHEME: {target_scheme} ===",
            "",
            "=== FORM FIELDS TO FILL (field name → type and constraints) ===",
            json.dumps(form_template, indent=2),
        ]

    return "\n".join(lines)


# ── Env HTTP calls ─────────────────────────────────────────────────────────

def env_reset(task_name: str, citizen_id: str, seed: int = 42) -> Dict:
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_name": task_name, "citizen_id": citizen_id, "seed": seed},
                      timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: Dict) -> Dict:
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Task runner ────────────────────────────────────────────────────────────

def run_task(llm: OpenAI, task_name: str) -> Dict[str, Any]:
    """Run one complete task episode. Returns score, rewards, steps, success."""
    cfg = TASK_CONFIG[task_name]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task_name, env="govscheme-env", model=MODEL_NAME)

    try:
        data = env_reset(task_name, cfg["citizen_id"], cfg["seed"])
        obs  = data.get("observation", data)

        for step in range(1, MAX_STEPS + 1):
            # Get action from LLM
            action_dict = call_llm(llm, task_name, obs)

            # Send action to environment
            try:
                result = env_step(action_dict)
            except Exception as e:
                error_msg = str(e)
                log_step(step, str(action_dict)[:100], 0.0, True, error_msg)
                break

            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", False))
            error_msg = result.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(action_dict)[:150],
                reward=reward,
                done=done,
                error=error_msg,
            )

            obs = result.get("observation", obs)

            if done:
                break

        # Score = best single-step score (agent gets 3 attempts)
        score   = max(rewards) if rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)
        if not rewards:
            rewards = [0.0]
        steps_taken = max(steps_taken, 1)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "rewards": rewards, "steps": steps_taken, "success": success}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"[DEBUG] ENV_URL   = {ENV_URL}",    flush=True)
    print(f"[DEBUG] MODEL     = {MODEL_NAME}", flush=True)
    print(f"[DEBUG] API_BASE  = {API_BASE_URL}", flush=True)

    # Health check
    try:
        h = requests.get(f"{ENV_URL}/health", timeout=10)
        print(f"[DEBUG] Env health: {h.json()}", flush=True)
    except Exception as e:
        print(f"[DEBUG] WARNING — env health check failed: {e}", flush=True)
        print("[DEBUG] Make sure the server is running before inference.py", flush=True)

    llm     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = []

    for task_name in TASKS:
        print(f"\n[DEBUG] ═══ Running task: {task_name} ═══", flush=True)
        result = run_task(llm, task_name)
        results.append(result)
        time.sleep(1)   # rate-limit politeness

    # Final summary
    print("\n[DEBUG] ═══ FINAL SCORES ═══", flush=True)
    avg = sum(r["score"] for r in results) / len(results)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"[DEBUG] {status} {r['task']}: score={r['score']:.3f}", flush=True)
    print(f"[DEBUG] Average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
