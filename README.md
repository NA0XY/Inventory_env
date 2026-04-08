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

Round 1 submission: a real-world OpenEnv benchmark for accounts-payable operations.

This environment models finance workflows that humans perform daily:

1. invoice field extraction from noisy OCR
2. invoice to purchase-order validation
3. fraud triage in invoice batches
4. GL coding for expense line items
5. vendor statement reconciliation

It is designed for agent learning and evaluation through the standard OpenEnv API:

1. reset()
2. step(action)
3. state()

## Round 1 Compliance Snapshot

| Requirement | Status | Notes |
|---|---|---|
| Real-world task (non-toy) | Pass | AP automation domain used in production finance workflows |
| OpenEnv API + typed models | Pass | Pydantic Observation/Action/Reward + reset/step/state implemented |
| 3+ tasks with graders | Pass | 5 tasks, deterministic graders, reward range [0.0, 1.0] |
| Meaningful reward shaping | Pass | Delta-based step rewards with partial progress signals |
| Baseline inference script | Pass | Root-level inference.py using OpenAI client and required env vars |
| HF Space + Docker deployable | Pass | Dockerized app on port 7860 and HF-ready metadata |
| Documentation completeness | Pass | Spaces, tasks, setup, validation, and submission checks documented |

## Real-World Utility

The agent acts as an AP analyst assistant. Given invoices and accounting context, it must:

1. **Extract** structured fields (vendor, invoice number, date, line items, amounts)
2. **Validate** invoices against purchase orders — detect quantity/price mismatches
3. **Detect fraud** in a batch — duplicates, unauthorized vendors, inflated amounts
4. **Assign GL codes** for ambiguous expense line items
5. **Reconcile statements** against internal ledgers for missing/discrepant invoices

## OpenEnv Interface

| Method | Endpoint | Behavior |
|---|---|---|
| reset(task_id, optional custom body) | POST /reset | Initializes task state and returns first observation |
| step(action) | POST /step | Applies one action and returns observation, reward, done, info |
| state() | GET /state | Returns current episode state snapshot |

All reward values are constrained to [0.0, 1.0].

## Typed Observation Space

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

## Typed Action Space

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

## Task Suite and Difficulty

| Task | Difficulty | Max Steps | Goal |
|---|---|---|---|
| `task_1` | Easy | 3 | Extract 7 fields from a noisy invoice |
| `task_2` | Medium | 3 | Validate invoice vs PO, find 3 mismatches |
| `task_3` | Hard | 3 | Detect 3 fraudulent invoices in a batch of 5 |
| `task_4` | Medium | 3 | Assign GL codes for 4 line items |
| `task_5` | Hard | 3 | Reconcile statement vs ledger |

## Grader and Reward Design

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

## Mandatory Inference Requirements

The root-level inference script is named inference.py and uses the OpenAI client.

Required environment variables:

1. API_BASE_URL
2. MODEL_NAME
3. HF_TOKEN

Recommended additional variable for local runs:

1. ENV_BASE_URL

The script emits strict structured stdout logs:

1. [START]
2. [STEP]
3. [END]

The formatting is enforced by a compliance checker in tools/check_inference_logs.py.

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

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset?task_id=task_1` | Start new episode |
| POST | `/step` | Submit action, get observation + reward |
| GET | `/state` | Read current episode state |

## Baseline Scores

Baseline run is deterministic at temperature 0 (model-dependent absolute scores).

| Task | Score |
|---|---|
| Task 1 — Field Extraction | task-dependent |
| Task 2 — PO Validation | task-dependent |
| Task 3 — Fraud Detection | task-dependent |
| Task 4 — GL Coding | task-dependent |
| Task 5 — Reconciliation | task-dependent |
| **Overall** | **computed at runtime** |

## Validation and Pre-Submission Checks

```bash
pip install openenv-core
openenv validate
```

```bash
# local container smoke
docker build -t invoice-env .
docker run -p 7860:7860 invoice-env
```

```bash
# strict inference log compliance
python tools/check_inference_logs.py --stdout stdout_check.log --stderr stderr_check.log
```

Submission gate checklist:

1. HF Space deploy responds with 200 and reset works
2. OpenEnv validation passes
3. Docker build and run succeed
4. Inference script completes and emits strict log format
5. Task graders produce valid [0,1] rewards

## Resource and Runtime Constraints

Target infra compatibility:

1. inference runtime under 20 minutes
2. compatible with vCPU=2 and memory=8GB

## HF Space Deployment

This repository is configured for Docker Spaces.

1. Push to a Hugging Face Space with hardware cpu-basic
2. Ensure Space tag includes openenv
3. App listens on port 7860 via Dockerfile runtime

## Judging Alignment (Round 1)

This submission is built to map directly to the published rubric:

1. Real-world utility: AP automation benchmark with realistic finance tasks
2. Task and grader quality: deterministic graders with easy to hard progression
3. Environment design: typed interfaces, delta rewards, clear episode boundaries
4. Code quality and compliance: OpenEnv-compatible structure + deployable container
5. Creativity and novelty: multi-stage AP flow in a single cohesive benchmark
