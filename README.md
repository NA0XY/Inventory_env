---
title: InvoiceProcessingEnv
emoji: "🧾"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - finance
---

# InvoiceProcessingEnv

An [OpenEnv](https://github.com/openenv) environment for training and evaluating
AI agents on **accounts-payable automation** — a real-world task performed by
finance operations teams at every company.

## What the agent does

The agent acts as an AP automation system. Given invoice text, it must:

1. **Extract** structured fields (vendor, invoice number, date, line items, amounts)
2. **Validate** invoices against purchase orders — detect quantity/price mismatches
3. **Detect fraud** in a batch — duplicates, unauthorized vendors, inflated amounts
4. **Assign GL codes** for ambiguous expense line items
5. **Reconcile statements** against internal ledgers for missing/discrepant invoices

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
| `chart_of_accounts` | object or null | GL mapping reference for task 4 |
| `vendor_statement` | string or null | Raw statement text for task 5 |
| `internal_ledger` | list or null | Internal ledger entries for task 5 |

## Action Space

| Field | Required by | Values |
|---|---|---|
| `invoice_id` | All | string |
| `extracted_fields` | Task 1 | dict of 7 fields |
| `decision` | Tasks 2 & 3 | `approve` / `reject` / `flag_for_review` |
| `mismatches` | Task 2 | list of strings |
| `fraud_flags` | Task 3 | list of `{invoice_id, reason}` |
| `gl_allocations` | Task 4 | object `{line_item_description: GL-XXXX}` |
| `missing_invoices` | Task 5 | list of invoice numbers |
| `discrepancy_invoices` | Task 5 | list of invoice numbers |

## Tasks

| Task | Difficulty | Max Steps | Goal |
|---|---|---|---|
| `task_1` | Easy | 3 | Extract 7 fields from a noisy invoice |
| `task_2` | Medium | 3 | Validate invoice vs PO, find 3 mismatches |
| `task_3` | Hard | 3 | Detect 3 fraudulent invoices in a batch of 5 |
| `task_4` | Medium | 3 | Assign GL codes for 4 line items |
| `task_5` | Hard | 3 | Reconcile statement vs ledger |

## Reward Design

**Task 1 (extraction):** `score = correct_fields / 7`
Each of the 7 fields contributes 1/7. Numeric fields tolerate ±$0.02 or ±2%.

**Task 2 (validation):** `0.4 × correct_decision + 0.2 × mismatches_found`
Max 1.0. Each of the 3 mismatches is detected via keyword matching.

**Task 3 (fraud detection):** `F1(precision, recall) + 0.05 × correct_reasons`
F1 across fraud IDs plus a reason bonus. Max 1.0.

**Task 4 (GL coding):** `0.25 × correct_gl_assignment`
One quarter point per correctly coded line item. Max 1.0.

**Task 5 (reconciliation):** `0.5 × F1(missing) + 0.5 × F1(discrepancy)`
Balances missing-invoice detection and discrepancy detection. Max 1.0.

Environment step rewards are **delta-based**: each step returns only improvement over the best score seen so far in the episode.

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
| Task 1 — Field Extraction | task-dependent |
| Task 2 — PO Validation | task-dependent |
| Task 3 — Fraud Detection | task-dependent |
| Task 4 — GL Coding | task-dependent |
| Task 5 — Reconciliation | task-dependent |
| **Overall** | **computed at runtime** |

## OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

## Inference Log Compliance Check

```bash
# after running inference with redirected streams
python tools/check_inference_logs.py --stdout stdout_check.log --stderr stderr_check.log
```

This verifies that stdout contains only strict `[START]`, `[STEP]`, and `[END]` lines,
and that no structured lines leak into stderr.

## HF Space Deployment

Push this repo to a Hugging Face Space with hardware `cpu-basic`.
The Dockerfile starts uvicorn on port 7860 automatically.
Tag the Space with `openenv`.
