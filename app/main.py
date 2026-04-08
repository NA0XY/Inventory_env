from __future__ import annotations

import os

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from typing import Any

from app.env import InvoiceProcessingEnv
from app.models import Action, Observation, ResetResult, StepResult, EnvState

app = FastAPI(title="InvoiceProcessingEnv", version="1.0.0")
_env = InvoiceProcessingEnv()


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    html_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Welcome to InvoiceProcessingEnv</h1><p>index.html not found. Make sure it is placed in the project root.</p>"


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
async def reset(
    task_id: str = Query(default="task_1"),
    custom_data: dict[str, Any] | None = Body(default=None),
) -> ResetResult:
    try:
        return _env.reset(task_id=task_id, custom_data=custom_data)
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
