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
            "Extract 7 fields from the invoice text. Fix OCR errors. "
            "Numbers must be floats. "
            "You have 3 attempts. Use environmental feedback to correct mistakes."
        ),
        difficulty="easy",
        max_steps=3,
    ),
    "task_2": TaskConfig(
        task_id="task_2",
        name="Purchase Order Validation",
        description=(
            "Compare invoice vs PO. Identify ALL mismatches. Verify Units of Measure. "
            "Set decision to: approve / reject / flag_for_review. "
            "List every mismatch in the mismatches field. "
            "You have 3 attempts. Use environmental feedback to self-correct."
        ),
        difficulty="medium",
        max_steps=3,
    ),
    "task_3": TaskConfig(
        task_id="task_3",
        name="Batch Fraud Detection",
        description=(
            "Review 5 invoices. Find frauds (duplicates, inflated amounts, unauthorized vendors). "
            "Do NOT flag legitimate invoices based on aggressive tone. "
            "You have 3 attempts. Use Precision/Recall feedback to adjust your flags."
        ),
        difficulty="hard",
        max_steps=3,
    ),
}

TASK_IDS = list(TASKS.keys())
