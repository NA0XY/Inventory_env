"""
FastAPI app exposing the OpenEnv interface.
Endpoints: POST /reset, POST /step, GET /state, GET /health, GET /tasks
"""

from __future__ import annotations
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from app.env import InvoiceProcessingEnv
from app.models import Action, Observation, ResetResult, StepResult, EnvState

app = FastAPI(
    title="InvoiceProcessingEnv",
    description="OpenEnv environment for AI-powered invoice processing and fraud detection.",
    version="1.0.0",
)

_env = InvoiceProcessingEnv()


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
        return """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>InvoiceProcessingEnv</title>
        <style>
            :root {
                --bg-1: #f6f8fb;
                --bg-2: #e9eff8;
                --ink: #152033;
                --muted: #5f6f86;
                --card: #ffffff;
                --line: #d3ddeb;
                --accent: #006d77;
                --accent-2: #0a9396;
            }

            * { box-sizing: border-box; }

            body {
                margin: 0;
                min-height: 100vh;
                font-family: "Segoe UI", "Trebuchet MS", Arial, sans-serif;
                color: var(--ink);
                background:
                    radial-gradient(circle at 15% 20%, #dce9ff 0%, transparent 34%),
                    radial-gradient(circle at 85% 10%, #d9f5f0 0%, transparent 35%),
                    linear-gradient(160deg, var(--bg-1), var(--bg-2));
            }

            .wrap {
                width: min(980px, 92vw);
                margin: 0 auto;
                padding: 40px 0 56px;
            }

            .hero {
                background: color-mix(in srgb, var(--card) 88%, #ffffff 12%);
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 28px;
                box-shadow: 0 12px 36px rgba(21, 32, 51, 0.1);
            }

            .kicker {
                display: inline-block;
                margin-bottom: 12px;
                padding: 6px 10px;
                border-radius: 999px;
                font-size: 12px;
                letter-spacing: 0.08em;
                font-weight: 700;
                text-transform: uppercase;
                color: #0b4f58;
                background: #d7f0f2;
            }

            h1 {
                margin: 0 0 10px;
                line-height: 1.1;
                font-size: clamp(28px, 4.5vw, 42px);
            }

            p {
                margin: 0;
                color: var(--muted);
                max-width: 72ch;
            }

            .grid {
                margin-top: 22px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 12px;
            }

            .tile {
                background: var(--card);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 14px;
            }

            .tile h3 {
                margin: 0 0 6px;
                font-size: 16px;
            }

            .tile p {
                font-size: 14px;
            }

            .links {
                margin-top: 22px;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }

            a.btn {
                text-decoration: none;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 700;
                transition: transform 120ms ease, box-shadow 120ms ease;
            }

            a.btn.primary {
                color: #ffffff;
                background: linear-gradient(90deg, var(--accent), var(--accent-2));
                box-shadow: 0 6px 20px rgba(10, 147, 150, 0.34);
            }

            a.btn.ghost {
                color: var(--ink);
                background: #ffffff;
                border: 1px solid var(--line);
            }

            a.btn:hover {
                transform: translateY(-1px);
            }

            .api {
                margin-top: 22px;
                border-radius: 14px;
                background: #ffffff;
                border: 1px solid var(--line);
                overflow: hidden;
            }

            .api table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }

            .api th,
            .api td {
                text-align: left;
                padding: 11px 14px;
                border-bottom: 1px solid var(--line);
            }

            .api tr:last-child td { border-bottom: 0; }

            .method {
                display: inline-block;
                min-width: 54px;
                text-align: center;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 800;
                padding: 3px 7px;
            }

            .get { background: #e8f8ef; color: #116636; }
            .post { background: #e6f2ff; color: #0d4c92; }
        </style>
    </head>
    <body>
        <main class="wrap">
            <section class="hero">
                <span class="kicker">OpenEnv Simulation</span>
                <h1>InvoiceProcessingEnv</h1>
                <p>
                    Accounts-payable environment for invoice field extraction, PO validation,
                    and fraud detection with deterministic grading.
                </p>

                <div class="grid">
                    <article class="tile">
                        <h3>Task 1: Extraction</h3>
                        <p>Extract 7 structured invoice fields from noisy OCR text.</p>
                    </article>
                    <article class="tile">
                        <h3>Task 2: PO Validation</h3>
                        <p>Compare invoice data against purchase order quantities, prices, and totals.</p>
                    </article>
                    <article class="tile">
                        <h3>Task 3: Fraud Detection</h3>
                        <p>Detect duplicate IDs, unauthorized vendors, and inflated amounts in batch review.</p>
                    </article>
                </div>

                <div class="links">
                    <a class="btn primary" href="/docs">Open API Docs</a>
                    <a class="btn ghost" href="/health">Health Check</a>
                    <a class="btn ghost" href="/tasks">List Tasks</a>
                </div>

                <section class="api" aria-label="API endpoints">
                    <table>
                        <thead>
                            <tr>
                                <th>Method</th>
                                <th>Endpoint</th>
                                <th>Purpose</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><span class="method get">GET</span></td>
                                <td>/health</td>
                                <td>Service health status</td>
                            </tr>
                            <tr>
                                <td><span class="method get">GET</span></td>
                                <td>/tasks</td>
                                <td>Task catalog and constraints</td>
                            </tr>
                            <tr>
                                <td><span class="method post">POST</span></td>
                                <td>/reset</td>
                                <td>Start task episode</td>
                            </tr>
                            <tr>
                                <td><span class="method post">POST</span></td>
                                <td>/step</td>
                                <td>Submit agent action and receive reward</td>
                            </tr>
                            <tr>
                                <td><span class="method get">GET</span></td>
                                <td>/state</td>
                                <td>Current environment state</td>
                            </tr>
                        </tbody>
                    </table>
                </section>
            </section>
        </main>
    </body>
</html>
"""


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "env": "invoice-processing-env", "version": "1.0.0"}


@app.get("/metadata")
async def metadata() -> dict[str, Any]:
    return {
        "name": "invoice-processing-env",
        "description": "OpenEnv environment for AI-powered invoice processing and fraud detection.",
        "version": "1.0.0",
    }


@app.get("/schema")
async def schema() -> dict[str, Any]:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EnvState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "result": {
            "status": "ok",
            "message": "InvoiceProcessingEnv MCP endpoint is reachable.",
        },
    }


@app.post("/reset", response_model=ResetResult)
async def reset(task_id: str = Query(default="task_1")) -> ResetResult:
    try:
        return _env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(action: Action) -> StepResult:
    try:
        return _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvState)
async def state() -> EnvState:
    return _env.state()


@app.get("/tasks")
async def list_tasks() -> dict:
    from app.tasks import TASKS
    return {
        tid: {
            "name": t.name,
            "description": t.description,
            "difficulty": t.difficulty,
            "max_steps": t.max_steps,
        }
        for tid, t in TASKS.items()
    }
