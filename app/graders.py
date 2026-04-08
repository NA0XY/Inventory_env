import re
from app.models import Action, Reward
from app.data import (
    TASK1_GROUND_TRUTH,
    TASK2_GROUND_TRUTH,
    TASK3_GROUND_TRUTH,
    TASK4_GROUND_TRUTH,
    TASK5_GROUND_TRUTH,
)


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
    failed_fields = [k for k, v in breakdown.items() if v < 1.0]
    feedback = "All fields correct!" if not failed_fields else f"SYSTEM REJECTION: Incorrect or missing fields: {', '.join(failed_fields)}. Please re-extract."
    return Reward(value=score, breakdown=breakdown, feedback=feedback)


def grade_task_2(action: Action) -> Reward:
    gt = TASK2_GROUND_TRUTH
    breakdown: dict[str, float] = {}

    got_dec = _normalise(action.decision or "")
    exp_dec = _normalise(gt["decision"])
    breakdown["decision"] = 0.4 if got_dec == exp_dec else 0.0

    got_mismatches = " ".join(_normalise(m) for m in (action.mismatches or []))
    rules = {
        "quantity": ["quantity", "600", "500", "pack", "uom"],
        "price": ["price", "14", "12"],
        "total": ["total", "2120", "1900", "5%", "exceed"],
    }

    matched_count = 0
    for name, words in rules.items():
        matched = sum(w in got_mismatches for w in words) >= 1
        breakdown[f"mismatch_{name}"] = 0.2 if matched else 0.0
        if matched:
            matched_count += 1

    total = min(sum(breakdown.values()), 1.0)
    if total >= 1.0:
        feedback = "Audit passed! Perfect validation."
    else:
        feedback = (
            f"SYSTEM REJECTION: Decision was {'CORRECT' if breakdown['decision'] > 0 else 'INCORRECT'}. "
            f"You found {matched_count}/3 mismatches. Check UoM math."
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
        if fid in gt["fraud_reasons"] and _normalise(gt["fraud_reasons"][fid]) in reason:
            reason_bonus += 0.05

    score = min(round(f1 + reason_bonus, 4), 1.0)
    breakdown = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reason_bonus": reason_bonus,
    }

    if score >= 1.0:
        feedback = "Batch Review Perfect!"
    else:
        feedback = (
            f"SYSTEM REJECTION: Precision: {precision:.2f}, Recall: {recall:.2f}. "
            f"False Positives: {fp}, Missed Frauds: {fn}. Adjust your flags."
        )
    return Reward(value=score, breakdown=breakdown, feedback=feedback)


def grade_task_4(action: Action) -> Reward:
    got_alloc = action.gl_allocations or {}
    gt = TASK4_GROUND_TRUTH
    breakdown: dict[str, float] = {}
    failed_items: list[str] = []

    for item, correct_gl in gt.items():
        got_gl = next(
            (
                v
                for k, v in got_alloc.items()
                if _normalise(k) in _normalise(item) or _normalise(item) in _normalise(k)
            ),
            "",
        )
        if str(got_gl).strip().upper() == correct_gl:
            breakdown[item] = 0.25
        else:
            breakdown[item] = 0.0
            failed_items.append(item)

    score = sum(breakdown.values())
    if score == 1.0:
        feedback = "All GL Codes perfectly allocated."
    else:
        feedback = f"SYSTEM REJECTION: Incorrect or missing GL codes for: {', '.join(failed_items)}."

    return Reward(value=round(score, 4), breakdown=breakdown, feedback=feedback)


def grade_task_5(action: Action) -> Reward:
    gt = TASK5_GROUND_TRUTH
    got_missing = set(action.missing_invoices or [])
    got_discrep = set(action.discrepancy_invoices or [])

    true_missing = set(gt["missing_invoices"])
    true_discrep = set(gt["discrepancy_invoices"])

    tp_m = len(got_missing & true_missing)
    f1_m = (2 * tp_m) / (len(got_missing) + len(true_missing)) if (len(got_missing) + len(true_missing)) > 0 else 0.0

    tp_d = len(got_discrep & true_discrep)
    f1_d = (2 * tp_d) / (len(got_discrep) + len(true_discrep)) if (len(got_discrep) + len(true_discrep)) > 0 else 0.0

    score = (f1_m * 0.5) + (f1_d * 0.5)

    if score == 1.0:
        feedback = "Reconciliation Perfect!"
    else:
        feedback = f"SYSTEM REJECTION: Missing F1: {f1_m:.2f}, Discrepancy F1: {f1_d:.2f}. Check the ledger closely."

    return Reward(
        value=round(score, 4),
        breakdown={"f1_missing": f1_m, "f1_discrepancy": f1_d},
        feedback=feedback,
    )
