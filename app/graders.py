"""
Deterministic graders for all three tasks.
All scores are in [0.0, 1.0].
"""

from __future__ import annotations
import re
from app.models import Action, Reward
from app.data import TASK1_GROUND_TRUTH, TASK2_GROUND_TRUTH, TASK3_GROUND_TRUTH


def _normalise(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _amount_close(a: float, b: float, tol: float = 0.02) -> bool:
    """True if two amounts are within 2 cents OR within 2%."""
    if b == 0:
        return abs(a) < 0.02
    return abs(a - b) / abs(b) <= tol or abs(a - b) <= 0.02


def grade_task_1(action: Action) -> Reward:
    """1/7 per correctly extracted field. Score = fraction correct."""
    ef = action.extracted_fields or {}
    gt = TASK1_GROUND_TRUTH
    breakdown: dict[str, float] = {}
    issues: list[str] = []

    # vendor_name
    got = _normalise(ef.get("vendor_name", ""))
    exp = _normalise(gt["vendor_name"])
    breakdown["vendor_name"] = 1.0 if got == exp else 0.0
    if not breakdown["vendor_name"]:
        issues.append(f"vendor_name: got '{got}', expected '{exp}'")

    # invoice_number
    got = _normalise(ef.get("invoice_number", ""))
    exp = _normalise(gt["invoice_number"])
    breakdown["invoice_number"] = 1.0 if got == exp else 0.0
    if not breakdown["invoice_number"]:
        issues.append(f"invoice_number: got '{got}', expected '{exp}'")

    # invoice_date
    got = _normalise(ef.get("invoice_date", ""))
    exp = _normalise(gt["invoice_date"])
    breakdown["invoice_date"] = 1.0 if got == exp else 0.0
    if not breakdown["invoice_date"]:
        issues.append(f"invoice_date: got '{got}', expected '{exp}'")

    # line_items count
    got_items = ef.get("line_items", [])
    exp_count = len(gt["line_items"])
    got_count = len(got_items) if isinstance(got_items, list) else 0
    if got_count == exp_count:
        breakdown["line_items"] = 1.0
    else:
        breakdown["line_items"] = max(0.0, 1.0 - abs(got_count - exp_count) / exp_count)
        issues.append(f"line_items count: got {got_count}, expected {exp_count}")

    # subtotal
    try:
        got_v = float(ef.get("subtotal", 0))
    except (TypeError, ValueError):
        got_v = 0.0
    breakdown["subtotal"] = 1.0 if _amount_close(got_v, gt["subtotal"]) else 0.0
    if not breakdown["subtotal"]:
        issues.append(f"subtotal: got {got_v}, expected {gt['subtotal']}")

    # tax_amount
    try:
        got_v = float(ef.get("tax_amount", 0))
    except (TypeError, ValueError):
        got_v = 0.0
    breakdown["tax_amount"] = 1.0 if _amount_close(got_v, gt["tax_amount"]) else 0.0
    if not breakdown["tax_amount"]:
        issues.append(f"tax_amount: got {got_v}, expected {gt['tax_amount']}")

    # total_amount
    try:
        got_v = float(ef.get("total_amount", 0))
    except (TypeError, ValueError):
        got_v = 0.0
    breakdown["total_amount"] = 1.0 if _amount_close(got_v, gt["total_amount"]) else 0.0
    if not breakdown["total_amount"]:
        issues.append(f"total_amount: got {got_v}, expected {gt['total_amount']}")

    score = round(sum(breakdown.values()) / 7.0, 4)
    feedback = "All fields correct!" if not issues else "Issues: " + "; ".join(issues)
    return Reward(value=score, breakdown=breakdown, feedback=feedback)


def grade_task_2(action: Action) -> Reward:
    """
    0.4 for correct decision.
    0.2 per correctly identified mismatch (max 3 mismatches = 0.6).
    Max total = 1.0.
    Mismatch matching uses keyword overlap (at least 2 of N keywords must appear).
    """
    gt = TASK2_GROUND_TRUTH
    breakdown: dict[str, float] = {}
    issues: list[str] = []

    # Decision
    got_dec = _normalise(action.decision or "")
    exp_dec = _normalise(gt["decision"])
    breakdown["decision"] = 0.4 if got_dec == exp_dec else 0.0
    if not breakdown["decision"]:
        issues.append(f"decision: got '{got_dec}', expected '{exp_dec}'")

    # Mismatch keywords - updated for UoM conversion logic
    KEYWORD_GROUPS = [
        ["cardboard", "box", "600", "500", "quantity", "pack", "50", "10"],
        ["bubble", "wrap", "14", "12", "unit price", "price"],
        ["total", "2120", "1900", "exceed", "5%"],
    ]
    got_mismatches = [_normalise(m) for m in (action.mismatches or [])]

    for i, keywords in enumerate(KEYWORD_GROUPS):
        matched = any(
            sum(kw in gm for kw in keywords) >= 2
            for gm in got_mismatches
        )
        score_i = 0.2 if matched else 0.0
        breakdown[f"mismatch_{i + 1}"] = score_i
        if not matched:
            issues.append(f"missed mismatch {i + 1}: {gt['mismatches'][i]}")

    total = min(sum(breakdown.values()), 1.0)
    feedback = "All correct!" if not issues else "; ".join(issues)
    return Reward(value=round(total, 4), breakdown=breakdown, feedback=feedback)


def grade_task_3(action: Action) -> Reward:
    """
    F1 score on fraud invoice IDs + 0.05 bonus per correct reason (max 0.15).
    Score capped at 1.0.
    """
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
            got_words = set(reason.split())
            if len(exp_words & got_words) >= 2:
                reason_bonus += 0.05

    score = min(round(f1 + reason_bonus, 4), 1.0)

    breakdown = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "reason_bonus": round(reason_bonus, 4),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
    }
    issues: list[str] = []
    if fp:
        issues.append(f"false positives: {list(flagged_ids - true_ids)}")
    if fn:
        issues.append(f"missed frauds: {list(true_ids - flagged_ids)}")
    feedback = "Perfect!" if not issues else "; ".join(issues)
    return Reward(value=score, breakdown=breakdown, feedback=feedback)
