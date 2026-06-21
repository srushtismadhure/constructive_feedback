"""Robot assignment coordinator.

Two modes behind one interface:
  - greedy: nearest idle eligible robot per pending task, no LLM
  - llm:    Claude with tool-use structured output, falls back to greedy on error
"""

from __future__ import annotations

import json
import math
import os
from typing import Literal

from orchestration.contracts import (
    Assignment,
    AssignmentBatch,
    RobotRegistry,
    SiteData,
    Task,
    TaskGraph,
)


# ---------------------------------------------------------------------------
# Greedy coordinator
# ---------------------------------------------------------------------------

def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def greedy_coordinator(task_graph: TaskGraph, registry: RobotRegistry) -> AssignmentBatch:
    """Assign each ready pending task to the nearest idle eligible robot."""
    done_ids = {t.task_id for t in task_graph.tasks if t.status == "done"}
    assigned_robot_ids: set[str] = set()
    assignments: list[Assignment] = []

    ready_tasks = [
        t for t in task_graph.tasks
        if t.status == "pending" and all(dep in done_ids for dep in t.depends_on)
    ]

    for task in ready_tasks:
        candidates = [
            r for r in registry.idle_robots(task.required_role)
            if r.id not in assigned_robot_ids
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: _manhattan(r.position, (task.position[0], task.position[1])))
        assignments.append(Assignment(robot_id=best.id, task_id=task.task_id))
        assigned_robot_ids.add(best.id)

    return AssignmentBatch(assignments=assignments, reasoning="greedy: nearest idle eligible robot")


# ---------------------------------------------------------------------------
# LLM coordinator (Claude)
# ---------------------------------------------------------------------------

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_MINIMAX_M3 = "accounts/fireworks/models/minimax-m3"

# Fireworks throttles bursty replan loops with HTTP 429. Retry with exponential
# backoff before the caller falls back to greedy.
_MAX_LLM_RETRIES = 4
_LLM_BACKOFF_BASE = 1.5  # seconds: 1.5, 3.0, 6.0, 12.0


def _create_with_retry(client, **kwargs):
    """Call chat.completions.create, retrying only on rate-limit (429)."""
    import time

    try:
        from openai import RateLimitError
    except ImportError:  # pragma: no cover — openai always present in this env
        RateLimitError = None

    last_exc: Exception | None = None
    for attempt in range(_MAX_LLM_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — re-raise non-429 immediately
            is_429 = (RateLimitError is not None and isinstance(exc, RateLimitError)) or (
                getattr(exc, "status_code", None) == 429
            )
            if not is_429 or attempt == _MAX_LLM_RETRIES - 1:
                raise
            last_exc = exc
            delay = _LLM_BACKOFF_BASE * (2 ** attempt)
            print(f"[coordinator] 429 rate-limited; retry {attempt + 1}/{_MAX_LLM_RETRIES - 1} in {delay:.1f}s")
            time.sleep(delay)
    if last_exc:  # pragma: no cover — loop always returns or raises above
        raise last_exc

# OpenAI-format tool definition (Fireworks is OpenAI-compatible)
_ASSIGN_TOOL = {
    "type": "function",
    "function": {
        "name": "assign_robots",
        "description": "Assign robots to tasks. Only include assignments you are confident about.",
        "parameters": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "robot_id": {"type": "string"},
                            "task_id": {"type": "string"},
                        },
                        "required": ["robot_id", "task_id"],
                    },
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of assignment decisions.",
                },
            },
            "required": ["assignments"],
        },
    },
}


def _build_prompt(
    ready_tasks: list[Task],
    registry: RobotRegistry,
    site: SiteData,
    context: str,
) -> str:
    task_lines = [
        f"  - {t.task_id}: {t.action} at {t.position}, needs {t.required_role}, "
        f"material={t.material or 'none'}, qty={t.quantity or 0}"
        for t in ready_tasks
    ]
    robot_lines = [
        f"  - {r.id} ({r.role}): pos={r.position}, status={r.status}"
        for r in registry.robots
        if r.status == "idle"
    ]
    hazard_lines = [f"  - {h}" for h in site.hazard_zones] if site.hazard_zones else ["  (none)"]

    parts = [
        "You are the robot coordinator for a Mars construction site.",
        "Assign idle robots to ready tasks. Only assign a robot if it has the correct role.",
        "Each robot can only take one task. Avoid sending robots near hazard zones if alternatives exist.",
        "",
        "READY TASKS:",
        *task_lines,
        "",
        "IDLE ROBOTS:",
        *robot_lines,
        "",
        "ACTIVE HAZARD ZONES (grid cells to avoid):",
        *hazard_lines,
    ]
    if context:
        parts += ["", f"CONTEXT: {context}"]

    parts += ["", "Call assign_robots with your assignments."]
    return "\n".join(parts)


def llm_coordinator(
    task_graph: TaskGraph,
    registry: RobotRegistry,
    site: SiteData,
    context: str = "",
    api_key: str | None = None,
) -> AssignmentBatch:
    """Call MiniMax M3 via Fireworks to assign robots. Falls back to greedy on any error.

    Fireworks is OpenAI-compatible — uses the openai client with a custom base_url.
    """
    try:
        from openai import OpenAI
    except ImportError:
        msg = "FALLBACK: openai SDK not installed"
        print(f"[coordinator] {msg}")
        result = greedy_coordinator(task_graph, registry)
        result.llm_fallback = True
        result.reasoning = msg
        return result

    done_ids = {t.task_id for t in task_graph.tasks if t.status == "done"}
    ready_tasks = [
        t for t in task_graph.tasks
        if t.status == "pending" and all(dep in done_ids for dep in t.depends_on)
    ]
    if not ready_tasks:
        return AssignmentBatch(assignments=[], reasoning="no ready tasks")

    eligible_roles = {t.required_role for t in ready_tasks}
    eligible_registry = RobotRegistry(
        robots=[r for r in registry.robots if r.status == "idle" and r.role in eligible_roles]
    )
    if not eligible_registry.robots:
        return AssignmentBatch(assignments=[], reasoning="no eligible idle robots")

    prompt = _build_prompt(ready_tasks, eligible_registry, site, context)
    key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not key:
        msg = "FALLBACK: FIREWORKS_API_KEY not set"
        print(f"[coordinator] {msg}")
        result = greedy_coordinator(task_graph, registry)
        result.llm_fallback = True
        result.reasoning = msg
        return result

    try:
        client = OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=key)
        response = _create_with_retry(
            client,
            model=_MINIMAX_M3,
            # MiniMax M3 emits reasoning before the tool call; 512 truncated the
            # arguments JSON mid-string (JSONDecodeError → silent greedy fallback).
            max_tokens=4096,
            temperature=0,
            tools=[_ASSIGN_TOOL],
            tool_choice={"type": "function", "function": {"name": "assign_robots"}},
            messages=[{"role": "user", "content": prompt}],
        )

        msg_obj = response.choices[0].message
        tool_calls = msg_obj.tool_calls or []
        call = next((tc for tc in tool_calls if tc.function.name == "assign_robots"), None)
        if call is None:
            raise ValueError("No assign_robots tool call in response")

        import json
        data = json.loads(call.function.arguments)
        assignments = [Assignment(**a) for a in data.get("assignments", [])]
        reasoning = data.get("reasoning", "")
        return AssignmentBatch(assignments=assignments, reasoning=f"[MiniMax M3] {reasoning}")

    except Exception as exc:
        msg = f"FALLBACK: LLM call failed ({type(exc).__name__}: {exc})"
        print(f"[coordinator] {msg}")
        result = greedy_coordinator(task_graph, registry)
        result.llm_fallback = True
        result.reasoning = msg
        return result


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def decide_assignments(
    task_graph: TaskGraph,
    registry: RobotRegistry,
    site: SiteData,
    context: str = "",
    mode: Literal["greedy", "llm"] = "llm",
    api_key: str | None = None,
) -> AssignmentBatch:
    if mode == "greedy":
        return greedy_coordinator(task_graph, registry)
    return llm_coordinator(task_graph, registry, site, context, api_key=api_key)
