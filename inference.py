import json
import os
import re
import sys
import time
from typing import Any, List, Optional

import httpx
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "invoice-processing-env"
MAX_STEPS = 3
SUCCESS_THRESHOLD = 0.6


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Any, reward: float, done: bool, error: Optional[str]):
    action_str = action if isinstance(action, str) else json.dumps(action)
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def call_llm(client: OpenAI, system: str, user: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=1500,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        return "{}"


def parse_json(text: str) -> dict:
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


def _history_to_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No previous attempts."
    lines = []
    for h in history:
        lines.append(
            f"Attempt {h['step']}: reward={h['reward']:.4f}, feedback={h['feedback']}, action={json.dumps(h['action'])}"
        )
    return "\n".join(lines)


def build_action(client: OpenAI, task_id: str, obs: dict, history: list[dict[str, Any]]) -> dict:
    history_text = _history_to_text(history)

    if task_id == "task_1":
        raw = call_llm(
            client,
            "You are an AP extraction specialist in a self-correction loop. Return ONLY JSON.",
            f"""Extract fields: vendor_name, invoice_number, invoice_date, line_items, subtotal, tax_amount, total_amount.
Invoice:\n{obs['invoice']['raw_text']}
History:\n{history_text}
Return ONLY JSON.""",
        )
        return {"invoice_id": obs["invoice"]["id"], "extracted_fields": parse_json(raw)}

    if task_id == "task_2":
        raw = call_llm(
            client,
            "You are an AP auditor in a self-correction loop. Return ONLY JSON.",
            f"""Compare invoice vs PO and output decision + mismatches.
Invoice:\n{obs['invoice']['raw_text']}
PO:\n{json.dumps(obs.get('purchase_order') or {}, indent=2)}
History:\n{history_text}
Return ONLY JSON with decision and mismatches.""",
        )
        parsed = parse_json(raw)
        return {
            "invoice_id": obs["invoice"]["id"],
            "decision": parsed.get("decision", "flag_for_review"),
            "mismatches": parsed.get("mismatches", []),
        }

    if task_id == "task_3":
        raw = call_llm(
            client,
            "You are an AP fraud investigator in a self-correction loop. Return ONLY JSON.",
            f"""Detect fraudulent invoices and provide fraud_flags.
Batch:\n{json.dumps(obs.get('batch') or [], indent=2)}
Whitelist:\n{json.dumps(obs.get('vendor_whitelist') or [], indent=2)}
History:\n{history_text}
Return ONLY JSON with fraud_flags.""",
        )
        parsed = parse_json(raw)
        first_id = (obs.get("batch") or [{"id": "unknown"}])[0]["id"]
        return {"invoice_id": first_id, "fraud_flags": parsed.get("fraud_flags", [])}

    if task_id == "task_4":
        raw = call_llm(
            client,
            "You are a finance GL coding assistant in a self-correction loop. Return ONLY JSON.",
            f"""Assign GL codes for each line item.
Invoice:\n{obs['invoice']['raw_text']}
Chart of Accounts:\n{json.dumps(obs.get('chart_of_accounts') or {}, indent=2)}
History:\n{history_text}
Return ONLY JSON with gl_allocations as {{line_item: GL-XXXX}}.""",
        )
        parsed = parse_json(raw)
        return {"invoice_id": obs["invoice"]["id"], "gl_allocations": parsed.get("gl_allocations", {})}

    raw = call_llm(
        client,
        "You are a reconciliation analyst in a self-correction loop. Return ONLY JSON.",
        f"""Find missing and discrepant invoices.
Vendor Statement:\n{obs.get('vendor_statement')}
Internal Ledger:\n{json.dumps(obs.get('internal_ledger') or [], indent=2)}
History:\n{history_text}
Return ONLY JSON with keys missing_invoices and discrepancy_invoices.""",
    )
    parsed = parse_json(raw)
    return {
        "invoice_id": "task5-batch",
        "missing_invoices": parsed.get("missing_invoices", []),
        "discrepancy_invoices": parsed.get("discrepancy_invoices", []),
    }


def run_task(client: OpenAI, http: httpx.Client, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    r = http.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()["observation"]

    rewards: list[float] = []
    history: list[dict[str, Any]] = []
    steps_taken = 0
    done = False

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        action = build_action(client, task_id, obs, history)

        try:
            sr = http.post(f"{ENV_BASE_URL}/step", json=action, timeout=30.0)
            sr.raise_for_status()
            result = sr.json()
        except Exception as exc:
            log_step(step=step, action=action, reward=0.0, done=True, error=str(exc))
            break

        reward = float(result["reward"]["value"])
        done = bool(result["done"])
        obs = result["observation"]
        feedback = result["reward"].get("feedback", "")

        rewards.append(reward)
        steps_taken = step
        history.append({"step": step, "action": action, "reward": reward, "feedback": feedback})

        log_step(step=step, action=action, reward=reward, done=done, error=None)

        if done:
            break

    score = round(sum(rewards), 4)
    score = min(max(score, 0.0), 1.0)
    log_end(success=score >= SUCCESS_THRESHOLD, steps=steps_taken, score=score, rewards=rewards)
    return score


def main() -> None:
    if not API_KEY:
        print("[ERROR] Set HF_TOKEN or OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    with httpx.Client(timeout=60.0) as http:
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
        for task_id in ["task_1", "task_2", "task_3", "task_4", "task_5"]:
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
