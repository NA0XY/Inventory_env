"""
Deterministic graders for all three tasks.
Provides directional feedback to enable agentic self-correction.
"""

from __future__ import annotations
import re
from app.models import Action, Reward
from app.data import TASK1_GROUND_TRUTH, TASK2_GROUND_TRUTH, TASK3_GROUND_TRUTH


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _amount_close(a: float, b: float, tol: float = 0.02) -> bool:
    if b == 0:
        return abs(a) < 0.02
    return abs(a - b) / abs(b) <= tol or abs(a - b) <= 0.02


def grade_task_1(action: Action) -> Reward:
    ef = action.extracted_fields or {}
    gt = TASK1_GROUND_TRUTH
    breakdown: dict[str, float] = {}

    breakdown["vendor_name"] = 1.0 if _normalise(ef.get("vendor_name", "")) == _normalise(gt["vendor_name"]) else 0.0
    breakdown["invoice_number"] = 1.0 if _normalise(ef.get("invoice_number", "")) == _normalise(gt["invoice_number"]) else 0.0
    breakdown["invoice_date"] = 1.0 if _normalise(ef.get("invoice_date", "")) == _normalise(gt["invoice_date"]) else 0.0

    got_items = ef.get("line_items", [])
    exp_count = len(gt["line_items"])
    got_count = len(got_items) if isinstance(got_items, list) else 0
    breakdown["line_items"] = 1.0 if got_count == exp_count else max(0.0, 1.0 - abs(got_count - exp_count) / exp_count)

    for field in ["subtotal", "tax_amount", "total_amount"]:
        try:
            val = float(ef.get(field, 0))
        except (TypeError, ValueError):
            val = 0.0
        breakdown[field] = 1.0 if _amount_close(val, gt[field]) else 0.0

    score = round(sum(breakdown.values()) / 7.0, 4)

    # Directional feedback (doesn't reveal correct values).
    failed_fields = [k for k, v in breakdown.items() if v < 1.0]
    feedback = "All fields correct!" if not failed_fields else f"SYSTEM REJECTION: Incorrect or missing fields: {', '.join(failed_fields)}. Please re-extract."

    return Reward(value=score, breakdown=breakdown, feedback=feedback)


def grade_task_2(action: Action) -> Reward:
    gt = TASK2_GROUND_TRUTH
    breakdown: dict[str, float] = {}

    got_dec = _normalise(action.decision or "")
    exp_dec = _normalise(gt["decision"])
    breakdown["decision"] = 0.4 if got_dec == exp_dec else 0.0

    KEYWORD_GROUPS = [
        ["cardboard", "box", "600", "500", "quantity", "pack", "50", "10"],
        ["bubble", "wrap", "14", "12", "unit price", "price"],
        ["total", "2120", "1900", "exceed", "5%"],
    ]
    got_mismatches = [_normalise(m) for m in (action.mismatches or [])]

    mismatches_found = 0
    for i, keywords in enumerate(KEYWORD_GROUPS):
        matched = any(sum(kw in gm for kw in keywords) >= 2 for gm in got_mismatches)
        breakdown[f"mismatch_{i + 1}"] = 0.2 if matched else 0.0
        if matched:
            mismatches_found += 1

    total = min(sum(breakdown.values()), 1.0)

    if total >= 1.0:
        feedback = "Audit passed! Perfect validation."
    else:
        feedback = (
            f"SYSTEM REJECTION: Decision was {'CORRECT' if breakdown['decision'] > 0 else 'INCORRECT'}. "
            f"You found {mismatches_found}/3 mismatches. Check UoM math."
        )

    return Reward(value=round(total, 4), breakdown=breakdown, feedback=feedback)


def grade_task_3(action: Action) -> Reward:
    gt = TASK3_GROUND_TRUTH
    flags = action.fraud_flags or []

    flagged_ids = {f.get("invoice_id", "") for f in flags}
    true_ids = set(gt["fraudulent_ids"])

    tp = len(flagged_ids & true_ids)
    fp = len(flagged_ids - true_ids)
    fn = len(true_ids - flagged_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    reason_bonus = 0.0
    for f in flags:
        fid = f.get("invoice_id", "")
        reason = _normalise(f.get("reason", ""))
        if fid in gt["fraud_reasons"]:
            exp_words = set(_normalise(gt["fraud_reasons"][fid]).split())
            if len(exp_words & set(reason.split())) >= 2:
                reason_bonus += 0.05

    score = min(round(f1 + reason_bonus, 4), 1.0)
    breakdown = {"precision": precision, "recall": recall, "f1": f1}

    if score >= 1.0:
        feedback = "Batch Review Perfect!"
    else:
        feedback = (
            f"SYSTEM REJECTION: Precision: {precision:.2f}, Recall: {recall:.2f}. "
            f"False Positives: {fp}, Missed Frauds: {fn}. Adjust your flags."
        )

    return Reward(value=score, breakdown=breakdown, feedback=feedback)
