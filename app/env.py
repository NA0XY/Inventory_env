from __future__ import annotations

from typing import Optional
from app.models import Observation, Action, Reward, StepResult, ResetResult, EnvState
from app.tasks import TASKS, TaskConfig
from app.graders import grade_task_1, grade_task_2, grade_task_3, grade_task_4, grade_task_5
from app.data import (
    TASK1_INVOICE,
    TASK2_INVOICE,
    TASK2_PO,
    TASK3_INVOICES,
    TASK3_VENDOR_WHITELIST,
    TASK4_INVOICE,
    TASK4_CHART_OF_ACCOUNTS,
    TASK5_STATEMENT,
    TASK5_LEDGER,
)


class InvoiceProcessingEnv:
    def __init__(self) -> None:
        self._task_config: Optional[TaskConfig] = None
        self._step_number: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._best_score: float = 0.0
        self._episode_rewards: list[float] = []

    def reset(self, task_id: str = "task_1") -> ResetResult:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")
        self._task_config = TASKS[task_id]
        self._step_number = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._best_score = 0.0
        self._episode_rewards = []
        return ResetResult(observation=self._make_observation(), info={"task": task_id})

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._task_config is None:
            raise RuntimeError("Call reset() before step().")

        self._step_number += 1
        tid = self._task_config.task_id

        if tid == "task_1":
            rew = grade_task_1(action)
        elif tid == "task_2":
            rew = grade_task_2(action)
        elif tid == "task_3":
            rew = grade_task_3(action)
        elif tid == "task_4":
            rew = grade_task_4(action)
        elif tid == "task_5":
            rew = grade_task_5(action)
        else:
            raise ValueError(f"Unknown task_id: {tid}")

        current_score = rew.value
        step_reward_val = max(0.0, current_score - self._best_score)
        self._best_score = max(self._best_score, current_score)

        if current_score >= 1.0 or self._step_number >= self._task_config.max_steps:
            self._done = True

        self._cumulative_reward = round(self._cumulative_reward + step_reward_val, 4)
        self._episode_rewards.append(step_reward_val)

        rew_obj = Reward(value=round(step_reward_val, 4), breakdown=rew.breakdown, feedback=rew.feedback)
        return StepResult(
            observation=self._make_observation(),
            reward=rew_obj,
            done=self._done,
            info={
                "step": self._step_number,
                "feedback": rew.feedback,
                "cumulative_reward": self._cumulative_reward,
            },
        )

    def state(self) -> EnvState:
        return EnvState(
            task_id=self._task_config.task_id if self._task_config else "none",
            step_number=self._step_number,
            total_steps=self._task_config.max_steps if self._task_config else 0,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            episode_rewards=self._episode_rewards,
        )

    def _make_observation(self) -> Observation:
        if self._task_config is None:
            raise RuntimeError("Task not set. Call reset() first.")

        tid = self._task_config.task_id
        base = {
            "task_id": tid,
            "task_description": self._task_config.description,
            "step_number": self._step_number,
            "total_steps": self._task_config.max_steps,
        }

        if tid == "task_1":
            return Observation(**base, invoice=TASK1_INVOICE)
        if tid == "task_2":
            return Observation(**base, invoice=TASK2_INVOICE, purchase_order=TASK2_PO)
        if tid == "task_3":
            return Observation(**base, invoice=TASK3_INVOICES[0], vendor_whitelist=TASK3_VENDOR_WHITELIST, batch=TASK3_INVOICES)
        if tid == "task_4":
            return Observation(**base, invoice=TASK4_INVOICE, chart_of_accounts=TASK4_CHART_OF_ACCOUNTS)
        if tid == "task_5":
            return Observation(**base, vendor_statement=TASK5_STATEMENT, internal_ledger=TASK5_LEDGER)

        raise ValueError(f"Unknown task_id: {tid}")
