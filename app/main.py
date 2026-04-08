"""
FastAPI app exposing the OpenEnv interface.
Endpoints: POST /reset, POST /step, GET /state, GET /health, GET /tasks
"""

from __future__ import annotations
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from app.env import InvoiceProcessingEnv
from app.models import Action, Observation, ResetResult, StepResult, EnvState

app = FastAPI(
    title="InvoiceProcessingEnv",
    description="OpenEnv environment for AI-powered invoice processing and fraud detection.",
    version="1.0.0",
)

_env = InvoiceProcessingEnv()


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
