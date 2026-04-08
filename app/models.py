from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total: float


class Invoice(BaseModel):
    id: str
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PurchaseOrder(BaseModel):
    po_number: str
    vendor_name: str
    line_items: list[LineItem]
    total_amount: float
    currency: str = "USD"
    issued_date: str
    valid_until: str


class Observation(BaseModel):
    task_id: str
    task_description: str
    step_number: int
    total_steps: int
    invoice: Optional[Invoice] = None
    purchase_order: Optional[PurchaseOrder] = None
    vendor_whitelist: Optional[list[str]] = None
    batch: Optional[list[Invoice]] = None
    chart_of_accounts: Optional[dict[str, str]] = None
    vendor_statement: Optional[str] = None
    internal_ledger: Optional[list[dict[str, Any]]] = None


class Action(BaseModel):
    invoice_id: str
    extracted_fields: Optional[dict[str, Any]] = None
    decision: Optional[str] = None
    mismatches: Optional[list[str]] = None
    fraud_flags: Optional[list[dict[str, str]]] = None
    gl_allocations: Optional[dict[str, str]] = None
    missing_invoices: Optional[list[str]] = None
    discrepancy_invoices: Optional[list[str]] = None


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: Observation
    info: dict[str, Any] = Field(default_factory=dict)


class EnvState(BaseModel):
    task_id: str
    step_number: int
    total_steps: int
    done: bool
    cumulative_reward: float
    episode_rewards: list[float]
