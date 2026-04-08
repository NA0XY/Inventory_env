# InvoiceProcessingEnv

An [OpenEnv](https://github.com/openenv) environment for training and evaluating
AI agents on **accounts-payable automation** — a real-world task performed by
finance operations teams at every company.

## What the agent does

The agent acts as an AP automation system. Given invoice text, it must:

1. **Extract** structured fields (vendor, invoice number, date, line items, amounts)
2. **Validate** invoices against purchase orders — detect quantity/price mismatches
3. **Detect fraud** in a batch — duplicates, unauthorized vendors, inflated amounts

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Active task |
| `task_description` | string | What the agent must do |
| `step_number` | int | Current step |
| `total_steps` | int | Episode length |
| `invoice` | object | `{id, raw_text, metadata}` |
| `purchase_order` | object or null | PO for task 2 |
| `vendor_whitelist` | list or null | Approved vendors for task 3 |
| `batch` | list or null | All 5 invoices for task 3 |

## Action Space

| Field | Required by | Values |
|---|---|---|
| `invoice_id` | All | string |
| `extracted_fields` | Task 1 | dict of 7 fields |
| `decision` | Tasks 2 & 3 | `approve` / `reject` / `flag_for_review` |
| `mismatches` | Task 2 | list of strings |
| `fraud_flags` | Task 3 | list of `{invoice_id, reason}` |

## Tasks

| Task | Difficulty | Max Steps | Goal |
|---|---|---|---|
| `task_1` | Easy | 1 | Extract 7 fields from a clean invoice |
| `task_2` | Medium | 3 | Validate invoice vs PO, find 3 mismatches, approve/reject |
| `task_3` | Hard | 5 | Detect 3 fraudulent invoices in a batch of 5 |

## Reward Design

**Task 1 (extraction):** `score = correct_fields / 7`
Each of the 7 fields contributes 1/7. Numeric fields tolerate ±$0.02 or ±2%.

**Task 2 (validation):** `0.4 × correct_decision + 0.2 × mismatches_found`
Max 1.0. Each of the 3 mismatches is detected via keyword matching.

**Task 3 (fraud detection):** `F1(precision, recall) + 0.05 × correct_reasons`
F1 across fraud IDs plus a reason bonus. Max 1.0.

All rewards are partial — the agent gets signal at every step, not just at the end.

## Setup

```bash
# Run locally
pip install -r requirements.txt
uvicorn app.main:app --port 7860
```

```bash
# Docker
docker build -t invoice-env .
docker run -p 7860:7860 invoice-env
```

```bash
# Run baseline inference
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset?task_id=task_1` | Start new episode |
| POST | `/step` | Submit action, get observation + reward |
| GET | `/state` | Read current episode state |

## Baseline Scores (gpt-4o-mini, temperature=0)

| Task | Score |
|---|---|
| Task 1 — Field Extraction | ~0.86 |
| Task 2 — PO Validation | ~0.70 |
| Task 3 — Fraud Detection | ~0.55 |
| **Overall** | **~0.70** |

## OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

## HF Space Deployment

Push this repo to a Hugging Face Space with hardware `cpu-basic`.
The Dockerfile starts uvicorn on port 7860 automatically.
Tag the Space with `openenv`.
