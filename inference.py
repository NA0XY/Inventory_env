"""
inference.py - Baseline inference script for InvoiceProcessingEnv.

Environment variables required:
    API_BASE_URL   LLM API base URL  (e.g. https://api.openai.com/v1)
    MODEL_NAME     Model identifier  (e.g. gpt-4o-mini)
    HF_TOKEN       API key           (fallback: OPENAI_API_KEY)
    ENV_BASE_URL   Env URL           (default: http://localhost:7860)

Emits structured logs: [START] [STEP] [END]
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
from typing import Any, List, Optional

import httpx
from openai import OpenAI

# -- Config -------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY: str = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "invoice-processing-env"
MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.6


# -- Log helpers (REQUIRED FORMAT - do not change field names) ----------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Any, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_str = action if isinstance(action, str) else json.dumps(action)
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# -- LLM helper ----------------------------------------------------------------

def call_llm(client: OpenAI, system: str, user: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        return "{}"


def parse_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


# -- Action builders -----------------------------------------------------------

def build_action_task1(client: OpenAI, obs: dict) -> dict:
    invoice_text = obs["invoice"]["raw_text"]
    raw = call_llm(
        client,
        system="You are an expert AP clerk. Extract invoice fields and return ONLY raw JSON. No markdown.",
        user=f"""Extract exactly these 7 fields and return as JSON:
- vendor_name (string)
- invoice_number (string)
- invoice_date (string, YYYY-MM-DD)
- line_items (list of {{description, quantity, unit_price, total}} - numbers as floats)
- subtotal (float)
- tax_amount (float)
- total_amount (float)

INVOICE:
{invoice_text}

Return ONLY a JSON object with these 7 keys. Numbers must be floats.""",
    )
    fields = parse_json(raw)
    return {"invoice_id": obs["invoice"]["id"], "extracted_fields": fields}


def build_action_task2(client: OpenAI, obs: dict) -> dict:
    invoice_text = obs["invoice"]["raw_text"]
    po = obs.get("purchase_order") or {}
    raw = call_llm(
        client,
        system="You are an AP auditor. Compare invoice vs PO and return ONLY raw JSON. No markdown.",
        user=f"""Compare the invoice against the PO. Return JSON with:
- decision: "approve" | "reject" | "flag_for_review"
  (reject if total exceeds PO by >5% or material mismatch exists)
- mismatches: list of strings describing each mismatch (quantities, prices, totals)

INVOICE:
{invoice_text}

PURCHASE ORDER:
{json.dumps(po, indent=2)}

Return ONLY JSON with keys "decision" and "mismatches".""",
    )
    parsed = parse_json(raw)
    return {
        "invoice_id": obs["invoice"]["id"],
        "decision": parsed.get("decision", "flag_for_review"),
        "mismatches": parsed.get("mismatches", []),
    }


def build_action_task3(client: OpenAI, obs: dict) -> dict:
    batch = obs.get("batch") or []
    whitelist = obs.get("vendor_whitelist") or []
    batch_text = "\n\n---\n\n".join(inv["raw_text"] for inv in batch)
    raw = call_llm(
        client,
        system="You are a forensic AP fraud investigator. Return ONLY raw JSON. No markdown.",
        user=f"""Review this batch of 5 invoices and identify fraudulent ones.

Fraud types:
1. Duplicate invoice number (same number appears more than once in batch)
2. Total amount exceeds referenced PO approved total by more than 5%%
3. Vendor not on the approved whitelist
4. Any other clear fraud indicator

Approved vendor whitelist: {json.dumps(whitelist)}

BATCH OF INVOICES:
{batch_text}

Return a JSON object:
{{
  "fraud_flags": [
    {{"invoice_id": "INV-BATCH-XXX", "reason": "short reason string"}},
    ...
  ]
}}

Include ONLY fraudulent invoices. Return ONLY JSON.""",
    )
    parsed = parse_json(raw)
    first_id = batch[0]["id"] if batch else "unknown"
    return {
        "invoice_id": first_id,
        "fraud_flags": parsed.get("fraud_flags", []),
    }


# -- Task runner ---------------------------------------------------------------

def run_task(client: OpenAI, http: httpx.Client, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # Reset environment
    r = http.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()["observation"]

    rewards: list[float] = []
    steps_taken = 0
    done = False

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        # Build action with LLM
        if task_id == "task_1":
            action = build_action_task1(client, obs)
        elif task_id == "task_2":
            action = build_action_task2(client, obs)
        else:
            action = build_action_task3(client, obs)

        # Submit to env
        try:
            sr = http.post(f"{ENV_BASE_URL}/step", json=action, timeout=30.0)
            sr.raise_for_status()
            result = sr.json()
        except Exception as exc:
            log_step(step=step, action=action, reward=0.0, done=True, error=str(exc))
            break

        reward = result["reward"]["value"]
        done = result["done"]
        obs = result["observation"]
        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=action, reward=reward, done=done, error=None)

        if done:
            break

    score = round(sum(rewards) / max(len(rewards), 1), 4)
    score = min(max(score, 0.0), 1.0)
    log_end(success=score >= SUCCESS_THRESHOLD, steps=steps_taken, score=score, rewards=rewards)
    return score


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("[ERROR] Set HF_TOKEN or OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    with httpx.Client(timeout=60.0) as http:
        # Wait for env to be ready
        for attempt in range(12):
            try:
                if http.get(f"{ENV_BASE_URL}/health").status_code == 200:
                    break
            except Exception:
                pass
            print(f"[DEBUG] Waiting for env... ({attempt + 1}/12)", file=sys.stderr, flush=True)
            time.sleep(5)
        else:
            print("[ERROR] Env did not start.", file=sys.stderr)
            sys.exit(1)

        print("[INFO] Env ready. Running tasks.", flush=True)

        scores: dict[str, float] = {}
        for task_id in ["task_1", "task_2", "task_3"]:
            print(f"\n{'=' * 60}\n[INFO] Running {task_id}", flush=True)
            try:
                scores[task_id] = run_task(client, http, task_id)
            except Exception as exc:
                print(f"[ERROR] {task_id} crashed: {exc}", file=sys.stderr)
                scores[task_id] = 0.0

        overall = round(sum(scores.values()) / len(scores), 4)
        print("\n" + "=" * 60, flush=True)
        print(json.dumps({"event": "SUMMARY", "scores": scores, "overall_score": overall}), flush=True)


if __name__ == "__main__":
    main()
