from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from typing import Any

from app.env import InvoiceProcessingEnv
from app.models import Action, Observation, ResetResult, StepResult, EnvState

app = FastAPI(title="InvoiceProcessingEnv", version="1.0.0")
_env = InvoiceProcessingEnv()
_repo_root = Path(__file__).resolve().parent.parent


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


@app.post("/validate-submission")
async def validate_submission() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    try:
        _env.reset(task_id="task_1")
        checks.append({"name": "HF Space is live and responds to /reset", "passed": True, "detail": "POST /reset succeeded for task_1"})
    except Exception as exc:
        checks.append({"name": "HF Space is live and responds to /reset", "passed": False, "detail": str(exc)})

    docker_path = shutil.which("docker")
    if not docker_path:
        checks.append({"name": "Docker build succeeded", "passed": False, "detail": "docker command not found in this environment"})
    else:
        try:
            build = subprocess.run(
                [docker_path, "build", str(_repo_root)],
                cwd=_repo_root,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            passed = build.returncode == 0
            detail = (build.stdout or build.stderr or "").strip()
            checks.append({
                "name": "Docker build succeeded",
                "passed": passed,
                "detail": detail[-1000:] if detail else "docker build completed",
            })
        except Exception as exc:
            checks.append({"name": "Docker build succeeded", "passed": False, "detail": str(exc)})

    openenv_path = shutil.which("openenv")
    if not openenv_path:
        checks.append({"name": "openenv validate passed", "passed": False, "detail": "openenv command not found in this environment"})
    else:
        try:
            validate = subprocess.run(
                [openenv_path, "validate"],
                cwd=_repo_root,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            passed = validate.returncode == 0
            detail = (validate.stdout or validate.stderr or "").strip()
            checks.append({
                "name": "openenv validate passed",
                "passed": passed,
                "detail": detail[-1000:] if detail else "openenv validate completed",
            })
        except Exception as exc:
            checks.append({"name": "openenv validate passed", "passed": False, "detail": str(exc)})

    all_passed = all(check["passed"] for check in checks)
    return {
        "all_passed": all_passed,
        "summary": "All 3/3 checks passed!" if all_passed else "One or more checks failed.",
        "checks": checks,
    }
