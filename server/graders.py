"""
graders.py — Deterministic graders for all 3 tasks.
All graders return float in [0.0, 1.0] with partial credit.
No randomness. Same input → same score every time.
"""
from __future__ import annotations
import re
import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Task 1: Scheme Identification (Easy)
# Agent must identify all schemes a citizen is eligible for.
# Grader: F1 score with recall weighted 2x (missing schemes worse than false positives)
# ---------------------------------------------------------------------------

def grade_scheme_identification(
    predicted_ids: List[str],
    gold_ids: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Score = weighted F1: recall weighted more than precision.
    Missing an eligible scheme (low recall) is worse than including an extra one.

    Returns (score, breakdown_dict)
    """
    if not gold_ids:
        # Edge case: citizen qualifies for nothing
        score = 1.0 if not predicted_ids else 0.0
        return score, {"recall": score, "precision": score, "f1": score}

    predicted_set = set(predicted_ids)
    gold_set = set(gold_ids)

    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Weighted F-beta (beta=1.5 → recall weighted more)
    beta = 1.5
    if precision + recall == 0:
        f_beta = 0.0
    else:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # Standard F1 for reference
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    breakdown = {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "f_beta_1_5": round(f_beta, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "missed_schemes": list(gold_set - predicted_set),
        "extra_schemes": list(predicted_set - gold_set),
    }

    return round(f_beta, 4), breakdown


# ---------------------------------------------------------------------------
# Task 2: Scheme Ranking (Medium)
# Agent ranks eligible schemes by benefit value with justification.
# Grader: NDCG@K + benefit_amount accuracy + reasoning coverage
# ---------------------------------------------------------------------------

def _dcg(ranked_ids: List[str], gold_ranked: List[str], k: int) -> float:
    """Discounted Cumulative Gain."""
    gold_relevance = {sid: len(gold_ranked) - i for i, sid in enumerate(gold_ranked)}
    dcg = 0.0
    for i, sid in enumerate(ranked_ids[:k]):
        rel = gold_relevance.get(sid, 0)
        dcg += rel / math.log2(i + 2)  # log2(position+1), positions start at 1
    return dcg


def _ndcg(ranked_ids: List[str], gold_ranked: List[str], k: int) -> float:
    """Normalized DCG@K."""
    ideal_dcg = _dcg(gold_ranked, gold_ranked, k)
    if ideal_dcg == 0:
        return 1.0
    return _dcg(ranked_ids, gold_ranked, k) / ideal_dcg


def grade_scheme_ranking(
    ranked_schemes_action: List[Dict[str, Any]],
    gold_ranked: List[Dict[str, Any]],
    citizen: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Score breakdown:
    - 50%: NDCG@5 of ranking order
    - 30%: Benefit amount accuracy (agent must cite correct INR amounts)
    - 20%: Reasoning quality (mentions key eligibility criteria)

    Returns (score, breakdown_dict)
    """
    if not gold_ranked:
        return 1.0, {"ndcg": 1.0, "benefit_accuracy": 1.0, "reasoning_quality": 1.0}

    # --- NDCG component ---
    predicted_ids = [r.get("scheme_id", "") for r in ranked_schemes_action]
    gold_ids = [r["scheme_id"] for r in gold_ranked]
    k = min(5, len(gold_ranked))
    ndcg_score = _ndcg(predicted_ids, gold_ids, k)

    # --- Benefit accuracy component ---
    gold_benefit_map = {r["scheme_id"]: r["annual_benefit_inr"] for r in gold_ranked}
    benefit_scores = []
    for item in ranked_schemes_action:
        sid = item.get("scheme_id", "")
        if sid not in gold_benefit_map:
            continue
        cited_benefit = item.get("benefit_inr", item.get("annual_benefit_inr", None))
        gold_benefit = gold_benefit_map[sid]
        if gold_benefit == 0:
            # Credit/loan schemes: benefit is access, not INR amount; skip
            continue
        if cited_benefit is None:
            benefit_scores.append(0.0)
        else:
            try:
                cited = float(cited_benefit)
                # Within 10% tolerance for rounding variations
                ratio = cited / gold_benefit if gold_benefit > 0 else 0.0
                benefit_scores.append(1.0 if 0.9 <= ratio <= 1.1 else 0.0)
            except (TypeError, ValueError):
                benefit_scores.append(0.0)

    benefit_accuracy = sum(benefit_scores) / len(benefit_scores) if benefit_scores else 0.5

    # --- Reasoning quality component ---
    # Check if agent's reasoning covers key citizen attributes
    all_reasoning = " ".join(
        str(item.get("reason", "")) for item in ranked_schemes_action
    ).lower()

    reasoning_keywords = []
    if citizen.get("caste") in ["SC", "ST", "OBC"]:
        reasoning_keywords.append(citizen["caste"].lower())
    if citizen.get("occupation") == "farmer":
        reasoning_keywords.extend(["farm", "agricultur", "land", "crop"])
    if citizen.get("annual_family_income_inr", 0) < 100000:
        reasoning_keywords.extend(["income", "₹", "lakh", "benefi"])
    reasoning_keywords.extend(["eligib", "qualif"])

    if reasoning_keywords:
        hits = sum(1 for kw in reasoning_keywords if kw in all_reasoning)
        reasoning_quality = min(1.0, hits / len(reasoning_keywords))
    else:
        reasoning_quality = 0.8  # neutral

    # --- Weighted composite ---
    score = (0.50 * ndcg_score) + (0.30 * benefit_accuracy) + (0.20 * reasoning_quality)

    breakdown = {
        "ndcg_at_5": round(ndcg_score, 4),
        "benefit_accuracy": round(benefit_accuracy, 4),
        "reasoning_quality": round(reasoning_quality, 4),
        "composite_score": round(score, 4),
    }

    return round(min(score, 1.0), 4), breakdown


# ---------------------------------------------------------------------------
# Task 3: Form Filling (Hard)
# Agent fills application form for top-ranked scheme.
# Grader: field accuracy + format validity + completeness - hallucination penalty
# ---------------------------------------------------------------------------

FIELD_VALIDATORS: Dict[str, Any] = {
    "aadhaar": re.compile(r"^[2-9][0-9]{11}$"),
    "ifsc": re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$"),
    "mobile": re.compile(r"^[6-9][0-9]{9}$"),
    "date_dd_mm_yyyy": re.compile(r"^\d{2}/\d{2}/\d{4}$"),
}


def _validate_field_format(field_name: str, value: Any, field_spec: Dict[str, Any]) -> bool:
    """Returns True if value passes format validation for this field type."""
    ftype = field_spec.get("type", "string")
    if value is None or value == "":
        return False

    if ftype == "aadhaar":
        return bool(FIELD_VALIDATORS["aadhaar"].match(str(value)))
    elif ftype == "ifsc":
        return bool(FIELD_VALIDATORS["ifsc"].match(str(value).upper()))
    elif ftype == "mobile":
        return bool(FIELD_VALIDATORS["mobile"].match(str(value)))
    elif ftype == "date":
        fmt = field_spec.get("format", "DD/MM/YYYY")
        if fmt == "DD/MM/YYYY":
            return bool(FIELD_VALIDATORS["date_dd_mm_yyyy"].match(str(value)))
        return True
    elif ftype == "enum":
        return str(value) in [str(v) for v in field_spec.get("values", [])]
    elif ftype == "int":
        try:
            v = int(value)
            if "min" in field_spec and v < field_spec["min"]:
                return False
            if "max" in field_spec and v > field_spec["max"]:
                return False
            return True
        except (TypeError, ValueError):
            return False
    elif ftype == "float":
        try:
            v = float(value)
            if "min" in field_spec and v < field_spec["min"]:
                return False
            if "max" in field_spec and v > field_spec["max"]:
                return False
            return True
        except (TypeError, ValueError):
            return False
    elif ftype == "string":
        min_len = field_spec.get("min_length", 1)
        max_len = field_spec.get("max_length", 99999)
        s = str(value)
        return min_len <= len(s) <= max_len
    return True


def _field_value_matches_citizen(
    field_name: str,
    agent_value: Any,
    citizen: Dict[str, Any]
) -> Optional[bool]:
    """
    Returns True/False if field has a known citizen ground truth.
    Returns None if field is not directly verifiable from citizen profile.
    """
    citizen_field_map = {
        "applicant_name": citizen.get("name"),
        "girl_child_name": citizen.get("name"),
        "aadhaar_number": citizen.get("aadhaar_number"),
        "date_of_birth": citizen.get("date_of_birth"),
        "girl_child_dob": citizen.get("date_of_birth"),
        "gender": citizen.get("gender"),
        "state": citizen.get("state"),
        "district": citizen.get("district"),
        "village": citizen.get("village"),
        "bank_account_number": citizen.get("bank_account_number"),
        "ifsc_code": citizen.get("ifsc_code"),
        "mobile_number": citizen.get("mobile_number"),
        "father_name": citizen.get("father_name"),
        "guardian_name": citizen.get("guardian_name"),
        "guardian_aadhaar": citizen.get("guardian_aadhaar"),
        "caste_certificate_number": citizen.get("caste_certificate_number"),
        "weaver_id": citizen.get("weaver_id"),
        "loom_type": citizen.get("loom_type"),
        "crop_type": citizen.get("crop_type"),
        "land_area_acres": citizen.get("land_ownership_acres"),
        "annual_family_income_inr": citizen.get("annual_family_income_inr"),
        "category": citizen.get("caste"),
        "family_size": citizen.get("family_size"),
        "institution_name": citizen.get("institution"),
        "course_name": citizen.get("course"),
        "class": citizen.get("class_studying"),
        "minority_community": citizen.get("minority_community"),
    }

    gold = citizen_field_map.get(field_name)
    if gold is None:
        return None  # Not verifiable from profile

    # Normalise comparison
    return str(agent_value).strip() == str(gold).strip()


def grade_form_filling(
    form_data: Dict[str, Any],
    scheme: Dict[str, Any],
    citizen: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Score breakdown:
    - 40%: Field accuracy (value matches citizen profile ground truth)
    - 30%: Format validity (regex/type/range checks)
    - 20%: Completeness (all required fields present)
    - 10%: Hallucination penalty (deducted for fabricated values)

    Returns (score, breakdown_dict)
    """
    form_template = scheme.get("application_fields", {})
    if not form_template:
        return 0.5, {"note": "no_form_template"}

    required_fields = [f for f, spec in form_template.items() if spec.get("required", False)]
    all_fields = list(form_template.keys())

    # --- Completeness ---
    present_required = [f for f in required_fields if f in form_data and form_data[f] not in (None, "")]
    completeness = len(present_required) / len(required_fields) if required_fields else 1.0

    # --- Format validity & field accuracy ---
    format_results = {}
    accuracy_results = {}
    hallucination_flags = []

    for field_name in all_fields:
        if field_name not in form_data:
            continue
        agent_value = form_data[field_name]
        field_spec = form_template[field_name]

        # Format check
        format_ok = _validate_field_format(field_name, agent_value, field_spec)
        format_results[field_name] = 1.0 if format_ok else 0.0

        # Accuracy check (verifiable fields only)
        match = _field_value_matches_citizen(field_name, agent_value, citizen)
        if match is not None:
            accuracy_results[field_name] = 1.0 if match else 0.0
            if not match and format_ok:
                # Format-valid but wrong value → likely hallucination
                hallucination_flags.append(field_name)

    format_score = sum(format_results.values()) / len(format_results) if format_results else 0.0
    accuracy_score = sum(accuracy_results.values()) / len(accuracy_results) if accuracy_results else 0.5

    # --- Hallucination penalty ---
    # Penalty is proportional to number of hallucinated fields vs verifiable fields
    verifiable_count = len(accuracy_results) if accuracy_results else 1
    hallucination_penalty = min(0.3, len(hallucination_flags) * 0.05)  # Max -0.30
    # This is stored as negative for reporting but applied as subtraction below

    # --- Composite ---
    score = (
        0.40 * accuracy_score
        + 0.30 * format_score
        + 0.20 * completeness
        - hallucination_penalty
    )
    score = round(max(0.0, min(1.0, score)), 4)

    breakdown = {
        "completeness": round(completeness, 4),
        "format_validity": round(format_score, 4),
        "field_accuracy": round(accuracy_score, 4),
        "hallucination_penalty": round(-hallucination_penalty, 4),
        "hallucinated_fields": hallucination_flags,
        "missing_required_fields": [f for f in required_fields if f not in form_data or form_data[f] in (None, "")],
        "format_results": format_results,
        "accuracy_results": accuracy_results,
    }

    return score, breakdown
