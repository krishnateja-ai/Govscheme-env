"""
govscheme_env — Government Scheme Eligibility Matching OpenEnv Environment.

Quick start:
    from govscheme_env import GovSchemeEnv, GovSchemeAction

    client = GovSchemeEnv(base_url="http://localhost:7860")
    obs    = client.reset(task_name="scheme_identification", citizen_id="CIT_001", seed=42)
    result = client.step(GovSchemeAction(action_type="identify_schemes", scheme_ids=["PM_KISAN"]))
    client.close()
"""
from client import GovSchemeEnv
from models import GovSchemeAction, GovSchemeObservation, GovSchemeState

__all__ = ["GovSchemeEnv", "GovSchemeAction", "GovSchemeObservation", "GovSchemeState"]
