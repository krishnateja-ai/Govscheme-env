"""
eligibility.py — Deterministic eligibility rule engine.
Given a citizen profile and scheme database, returns exactly which schemes
the citizen qualifies for. This is the ground-truth oracle for Task 1 grading.
"""
from __future__ import annotations
from typing import Any, Dict, List


def check_eligibility(citizen: Dict[str, Any], scheme: Dict[str, Any]) -> bool:
    """
    Returns True if citizen meets ALL eligibility criteria for the scheme.
    Every rule is deterministic and auditable.
    """
    e = scheme.get("eligibility", {})

    # --- Occupation check ---
    allowed_occupations = e.get("occupation")
    if allowed_occupations is not None:
        if citizen.get("occupation") not in allowed_occupations:
            return False

    # --- Age check ---
    age_min = e.get("age_min")
    age_max = e.get("age_max")
    citizen_age = citizen.get("age", 0)
    if age_min is not None and citizen_age < age_min:
        return False
    if age_max is not None and citizen_age > age_max:
        return False

    # --- Gender check ---
    required_gender = e.get("gender")
    if required_gender is not None:
        if citizen.get("gender") != required_gender:
            return False

    # --- Caste check ---
    allowed_castes = e.get("caste_categories")
    if allowed_castes is not None:
        if citizen.get("caste") not in allowed_castes:
            return False

    # --- Minority community check ---
    allowed_minorities = e.get("minority_community")
    if allowed_minorities is not None:
        if citizen.get("minority_community") not in allowed_minorities:
            return False

    # --- Income checks ---
    income_max = e.get("income_annual_inr_max")
    if income_max is not None:
        if citizen.get("annual_income_inr", 0) > income_max:
            return False

    family_income_max = e.get("family_income_inr_max")
    if family_income_max is not None:
        if citizen.get("annual_family_income_inr", 0) > family_income_max:
            return False

    # --- Land checks ---
    land_min = e.get("land_ownership_acres_min")
    land_max = e.get("land_ownership_acres_max")
    citizen_land = citizen.get("land_ownership_acres", 0)
    if land_min is not None and citizen_land < land_min:
        return False
    if land_max is not None and citizen_land > land_max:
        return False

    # --- Document/infrastructure checks ---
    if e.get("requires_bank_account") and not citizen.get("has_bank_account", False):
        return False
    if e.get("requires_aadhaar") and not citizen.get("has_aadhaar", False):
        return False

    # --- Exclusion flags ---
    if e.get("exclude_govt_employee") and citizen.get("is_govt_employee", False):
        return False
    if e.get("exclude_income_taxpayer") and citizen.get("is_income_taxpayer", False):
        return False
    if e.get("exclude_professional") and citizen.get("is_professional", False):
        return False

    # --- Rural-only schemes ---
    if e.get("rural_only") and citizen.get("area_type") != "rural":
        return False

    # --- Housing condition ---
    if e.get("houseless_or_kachha"):
        if citizen.get("house_type") not in ["kachha", "houseless", "damaged"]:
            return False

    # --- LPG condition ---
    if e.get("no_existing_lpg") and citizen.get("has_lpg", True):
        return False

    # --- Education level ---
    edu_level = e.get("education_level")
    if edu_level is not None:
        citizen_edu = citizen.get("education", "")
        edu_hierarchy = {
            "pre_matric": ["pre_matric", "5th_pass", "8th_pass"],
            "post_matric": ["post_matric", "10th_pass", "12th_pass", "graduate", "post_graduate"]
        }
        valid_edu = edu_hierarchy.get(edu_level, [edu_level])
        if citizen_edu not in valid_edu:
            return False

    # --- Guardian requirement (for minor-focused schemes) ---
    if e.get("guardian_required"):
        if not citizen.get("guardian_name") or not citizen.get("guardian_aadhaar"):
            return False

    # --- Stand-Up India special rule: SC/ST OR Female ---
    sid = scheme.get("scheme_id", "")
    if sid == "STAND_UP_INDIA":
        is_sc_st = citizen.get("caste") in ["SC", "ST"]
        is_female = citizen.get("gender") == "Female"
        if not (is_sc_st or is_female):
            return False

    return True


def get_eligible_schemes(citizen: Dict[str, Any], all_schemes: List[Dict[str, Any]]) -> List[str]:
    """Returns list of scheme_ids the citizen is eligible for."""
    return [s["scheme_id"] for s in all_schemes if check_eligibility(citizen, s)]


def rank_schemes_by_benefit(
    eligible_scheme_ids: List[str],
    all_schemes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Rank eligible schemes by annual_benefit_inr descending.
    Returns list of {scheme_id, name, annual_benefit_inr, benefit_description} dicts.
    Schemes with benefit=0 (credit/loan schemes) go to the bottom, ranked by scheme name.
    """
    scheme_map = {s["scheme_id"]: s for s in all_schemes}
    eligible = [scheme_map[sid] for sid in eligible_scheme_ids if sid in scheme_map]

    # Separate into monetary benefit vs access-based
    monetary = [s for s in eligible if s["annual_benefit_inr"] > 0]
    access = [s for s in eligible if s["annual_benefit_inr"] == 0]

    monetary_sorted = sorted(monetary, key=lambda s: s["annual_benefit_inr"], reverse=True)
    access_sorted = sorted(access, key=lambda s: s["name"])

    ranked = monetary_sorted + access_sorted
    return [
        {
            "rank": i + 1,
            "scheme_id": s["scheme_id"],
            "name": s["name"],
            "annual_benefit_inr": s["annual_benefit_inr"],
            "benefit_description": s["benefit_description"]
        }
        for i, s in enumerate(ranked)
    ]
