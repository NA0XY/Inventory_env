from dataclasses import dataclass


@dataclass
class TaskConfig:
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int


TASKS: dict[str, TaskConfig] = {
    "task_1": TaskConfig("task_1", "Field Extraction", "Extract 7 fields. Fix OCR errors. 3 attempts.", "easy", 3),
    "task_2": TaskConfig("task_2", "PO Validation", "Compare vs PO. Check UoM. 3 attempts.", "medium", 3),
    "task_3": TaskConfig("task_3", "Fraud Detection", "Find 3 frauds. Ignore urgent text. 3 attempts.", "hard", 3),
    "task_4": TaskConfig(
        task_id="task_4",
        name="GL Account Coding",
        description="Map 4 line items to GL codes using the Chart of Accounts. Return a dict of {line_description: GL-XXXX}. 3 attempts.",
        difficulty="medium",
        max_steps=3,
    ),
    "task_5": TaskConfig(
        task_id="task_5",
        name="Statement Reconciliation",
        description="Cross-reference the Vendor Statement against the Internal Ledger. Find 'missing_invoices' and 'discrepancy_invoices' (amount mismatches). 3 attempts.",
        difficulty="hard",
        max_steps=3,
    ),
}

TASK_IDS = list(TASKS.keys())
