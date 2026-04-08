from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from typing import Any

from app.env import InvoiceProcessingEnv
from app.models import Action, Observation, ResetResult, StepResult, EnvState

app = FastAPI(title="InvoiceProcessingEnv", version="1.0.0")
_env = InvoiceProcessingEnv()


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return r"""
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>InvoiceProcessingEnv</title>
        <style>
            :root { --bg-1: #0f172a; --bg-2: #1e293b; --ink: #f8fafc; --card: #1e293b; --line: #334155; --accent: #38bdf8; }
            body { margin: 0; font-family: system-ui, sans-serif; color: var(--ink); background: linear-gradient(135deg, var(--bg-1), var(--bg-2)); min-height: 100vh; }
            .wrap { width: min(1000px, 92vw); margin: 0 auto; padding: 40px 0; }
            .hero { background: rgba(30,41,59,0.7); border: 1px solid var(--line); border-radius: 24px; padding: 32px; backdrop-filter: blur(12px); box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); }
            .kicker { display: inline-block; padding: 6px 12px; border-radius: 99px; font-size: 12px; font-weight: bold; background: #0ea5e920; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }
            h1 { margin: 0 0 12px; font-size: 2.5rem; }
            p { color: #94a3b8; font-size: 1.1rem; line-height: 1.6; max-width: 600px; margin-top:0; }

            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 32px 0; }
            .tile { background: #0f172a; border: 1px solid var(--line); border-radius: 16px; padding: 20px; transition: transform 0.2s; }
            .tile:hover { transform: translateY(-4px); border-color: var(--accent); }
            .tile h3 { margin: 0 0 8px; color: #e2e8f0; display: flex; align-items: center; gap: 8px; }
            .tile h3 span { background: #334155; padding: 2px 8px; border-radius: 6px; font-size: 12px; color: #94a3b8; }
            .tile p { font-size: 0.9rem; color: #64748b; margin: 0; }

            .btn { text-decoration: none; border-radius: 8px; padding: 10px 16px; font-weight: 600; cursor: pointer; border: none; font-size: 14px; transition: all 0.2s; }
            .btn-primary { background: var(--accent); color: #0f172a; }
            .btn-primary:hover { background: #7dd3fc; }
            .btn-ghost { background: transparent; color: #e2e8f0; border: 1px solid var(--line); }
            .btn-ghost:hover { background: #334155; }
            .actions { display: flex; gap: 12px; flex-wrap: wrap; }

            /* Syntax Highlighted JSON Viewer */
            #json-viewer { display: none; margin-top: 24px; background: #020617; border-radius: 16px; border: 1px solid var(--line); overflow: hidden; }
            .jv-header { display: flex; justify-content: space-between; padding: 12px 20px; background: #0f172a; border-bottom: 1px solid var(--line); align-items: center; }
            .jv-header h3 { margin: 0; font-family: monospace; color: var(--accent); font-size: 14px; }
            .jv-close { background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 18px; }
            .jv-close:hover { color: #fff; }
            pre { margin: 0; padding: 20px; overflow-x: auto; font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.5; }
            .string { color: #a7f3d0; } .number { color: #fde047; } .boolean { color: #f472b6; } .null { color: #94a3b8; } .key { color: #7dd3fc; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="wrap"><div class="hero">
            <div class="kicker">OpenEnv Agent Benchmark</div>
            <h1>InvoiceProcessingEnv</h1>
            <p>5-Stage multi-turn simulation testing OCR extraction, UoM Math, Fraud detection, GL Account Coding, and Ledger Reconciliation.</p>

            <div class="grid">
                <div class="tile"><h3>Task 1 <span>Extraction</span></h3><p>Fix OCR noise and extract 7 fields.</p></div>
                <div class="tile"><h3>Task 2 <span>Validation</span></h3><p>Verify quantities using Unit-of-Measure math.</p></div>
                <div class="tile"><h3>Task 3 <span>Fraud</span></h3><p>Batch review with Poison Pill distractors.</p></div>
                <div class="tile"><h3>Task 4 <span>GL Coding</span></h3><p>Map items to the Chart of Accounts.</p></div>
                <div class="tile"><h3>Task 5 <span>Reconciliation</span></h3><p>Cross-reference Statements vs Ledgers.</p></div>
            </div>

            <div class="actions">
                <a href="/docs" class="btn btn-primary">Swagger UI</a>
                <button class="btn btn-ghost" onclick="fetchEndpoint('/tasks')">View Tasks</button>
                <button class="btn btn-ghost" onclick="fetchEndpoint('/state')">Env State</button>
                <button class="btn btn-ghost" onclick="fetchEndpoint('/schema')">Schema</button>
            </div>

            <div id="json-viewer">
                <div class="jv-header"><h3 id="jv-title"></h3><button class="jv-close" onclick="document.getElementById('json-viewer').style.display='none'">x</button></div>
                <pre><code id="jv-content"></code></pre>
            </div>
        </div></div>

        <script>
            function syntaxHighlight(json) {
                json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
                    var cls = 'number';
                    if (/^"/.test(match)) {
                        if (/:$/.test(match)) { cls = 'key'; } else { cls = 'string'; }
                    } else if (/true|false/.test(match)) { cls = 'boolean'; } else if (/null/.test(match)) { cls = 'null'; }
                    return '<span class="' + cls + '">' + match + '</span>';
                });
            }
            async function fetchEndpoint(path) {
                document.getElementById('json-viewer').style.display = 'block';
                document.getElementById('jv-title').innerText = 'GET ' + path;
                document.getElementById('jv-content').innerHTML = '<span style="color:#94a3b8">Loading...</span>';
                try {
                    const r = await fetch(path); const d = await r.json();
                    document.getElementById('jv-content').innerHTML = syntaxHighlight(JSON.stringify(d, null, 2));
                } catch(e) { document.getElementById('jv-content').innerText = String(e); }
            }
        </script>
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
        "result": {"status": "ok", "message": "InvoiceProcessingEnv MCP endpoint is reachable."},
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
