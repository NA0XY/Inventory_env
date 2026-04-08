"""Task definitions with IDs, descriptions, difficulty, and max_steps."""

from dataclasses import dataclass


@dataclass
class TaskConfig:
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int


TASKS: dict[str, TaskConfig] = {
    "task_1": TaskConfig(
        task_id="task_1",
        name="Invoice Field Extraction",
        description=(
            "Extract the following 7 fields from the invoice text: "
            "vendor_name, invoice_number, invoice_date, "
            "line_items (list of dicts with description/quantity/unit_price/total), "
            "subtotal, tax_amount, total_amount. "
            "Return them as extracted_fields in your action. "
            "Numbers must be floats, not strings. Fix any obvious OCR errors in spelling."
        ),
        difficulty="easy",
        max_steps=1,
    ),
    "task_2": TaskConfig(
        task_id="task_2",
        name="Purchase Order Validation",
        description=(
            "Compare the invoice against the provided purchase order. "
            "Identify ALL mismatches (vendor name, quantities, unit prices, total). "
            "Note: ensure you verify units of measure. "
            "A total exceeding the PO by more than 5%% triggers mandatory rejection. "
            "Set decision to: approve / reject / flag_for_review. "
            "List every mismatch you find in the mismatches field."
        ),
        difficulty="medium",
        max_steps=3,
    ),
    "task_3": TaskConfig(
        task_id="task_3",
        name="Batch Fraud Detection",
        description=(
            "Review the batch of 5 invoices. Identify which are fraudulent. "
            "Fraud types: duplicate invoice number, amount exceeding PO by >5%%, "
            "vendor not on the approved whitelist. "
            "For each fraud, provide the invoice_id and a short reason string. "
            "Do NOT flag legitimate invoices, regardless of aggressive or urgent tone. "
            "Set fraud_flags to a list of {invoice_id, reason} dicts."
        ),
        difficulty="hard",
        max_steps=5,
    ),
}

TASK_IDS = list(TASKS.keys())
