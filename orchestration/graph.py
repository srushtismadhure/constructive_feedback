"""LangGraph state machine: the orchestration brain.

Flow:
  init → coordinate → validate →(pass)→ dispatch → monitor → check_done
                              ↗(fail, retry<3)                    |
                     coordinate ← replan ←──────────(trigger)────┘
                                                ↓(all done)
                                               END
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from orchestration.blueprint_parser import parse_blueprint
from orchestration.contracts import (
    Action,
    OrchestrationState,
    RobotRegistry,
    SiteData,
    StatusReport,
)
from orchestration.coordinator import decide_assignments
from orchestration.env_interface import EnvInterface
from orchestration.monitor import process_status
from orchestration.sequencer import sequence
from orchestration.validator import validate_assignments

_MAX_VALIDATE_RETRIES = 3
_MAX_STEPS = 200


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_init(state: OrchestrationState) -> dict:
    blueprint = parse_blueprint(state["blueprint_id"])
    task_graph = sequence(blueprint)

    # Build a minimal flat SiteData if the env hasn't provided one yet
    site = state.get("site") or SiteData(
        terrain=[[0.0] * 8 for _ in range(8)],
        buildable_mask=[[True] * 8 for _ in range(8)],
    )

    log = state.get("event_log", [])
    log.append(f"[init] blueprint '{blueprint.building_id}' parsed: {len(task_graph.tasks)} tasks")

    return {
        "blueprint": blueprint,
        "task_graph": task_graph,
        "site": site,
        "validate_retries": 0,
        "done": False,
        "event_log": log,
    }


def node_coordinate(state: OrchestrationState) -> dict:
    task_graph = state["task_graph"]
    registry = state["registry"]
    site = state["site"]
    trigger = state.get("last_trigger")

    context = ""
    if trigger:
        task_graph_tasks = {t.task_id: t for t in task_graph.tasks}
        failed = task_graph_tasks.get(trigger.task_id)
        if failed:
            failed.status = "pending"  # reset so it can be reassigned
        context = f"Replan: task '{trigger.task_id}' failed ({trigger.reason}). Reassign."

    batch = decide_assignments(
        task_graph=task_graph,
        registry=registry,
        site=site,
        context=context,
        mode=state.get("coordinator_mode", "llm"),
    )

    log = state.get("event_log", [])
    log.append(f"[coordinate] {len(batch.assignments)} assignments. reasoning: {batch.reasoning or '—'}")
    if batch.llm_fallback:
        log.append(f"[coordinate] WARNING: {batch.reasoning}")

    return {
        "pending_assignments": batch,
        "last_trigger": None,
        "event_log": log,
    }


def node_validate(state: OrchestrationState) -> dict:
    batch = state["pending_assignments"]
    task_graph = state["task_graph"]
    registry = state["registry"]
    inventory = state.get("inventory", {})
    retries = state.get("validate_retries", 0)

    errors = validate_assignments(batch, task_graph, registry, inventory)
    log = state.get("event_log", [])

    if errors:
        log.append(f"[validate] FAIL (retry {retries+1}): {errors}")
        # filter out bad assignments and keep valid ones
        bad_task_ids = set()
        task_map = {t.task_id: t for t in task_graph.tasks}
        robot_map = {r.id: r for r in registry.robots}
        done_ids = {t.task_id for t in task_graph.tasks if t.status == "done"}
        valid_assignments = []
        for a in batch.assignments:
            r = robot_map.get(a.robot_id)
            t = task_map.get(a.task_id)
            if (
                r and t
                and r.role == t.required_role
                and r.status == "idle"
                and all(dep in done_ids for dep in t.depends_on)
            ):
                valid_assignments.append(a)
        from orchestration.contracts import AssignmentBatch
        return {
            "pending_assignments": AssignmentBatch(
                assignments=valid_assignments,
                reasoning=batch.reasoning,
            ),
            "validate_retries": retries + 1,
            "event_log": log,
        }

    log.append(f"[validate] PASS: {len(batch.assignments)} assignments")
    return {"validate_retries": 0, "event_log": log}


def node_dispatch(state: OrchestrationState, env: EnvInterface) -> dict:
    """Send each assignment to the sim as an Action, mark tasks in_progress."""
    batch = state["pending_assignments"]
    task_graph = state["task_graph"]
    registry = state["registry"]
    inventory = dict(state.get("inventory", {}))
    log = state.get("event_log", [])
    step = state.get("step", 0)

    task_map = {t.task_id: t for t in task_graph.tasks}
    robot_map = {r.id: r for r in registry.robots}

    status_reports: list[StatusReport] = []

    for assignment in batch.assignments:
        task = task_map.get(assignment.task_id)
        robot = robot_map.get(assignment.robot_id)
        if not task or not robot:
            continue

        task.status = "in_progress"
        robot.status = "busy"

        _CMD_MAP = {"haul": "pickup", "excavate": "excavate", "place": "place", "weld": "weld"}
        action = Action(
            robot_id=robot.id,
            command=_CMD_MAP.get(task.action, "noop"),  # type: ignore[arg-type]
            target=task.position,
            material=task.material,
        )

        obs, reward, done_env, info = env.step(action)
        step += 1

        rejection = info.get("rejection_reason")
        if rejection:
            result = "failed"
            reason = rejection
            log.append(f"[dispatch] {robot.id}→{task.task_id} REJECTED: {reason}")
        else:
            result = "complete"
            reason = None
            log.append(f"[dispatch] {robot.id}→{task.task_id} OK (reward={reward:.2f})")

        status_reports.append(
            StatusReport(
                robot_id=robot.id,
                task_id=task.task_id,
                result=result,
                reason=reason,
                new_position=(robot.position[0], robot.position[1]),
                resources_consumed={task.material: task.quantity}
                if result == "complete" and task.material and task.quantity
                else {},
            )
        )

    return {
        "task_graph": task_graph,
        "registry": registry,
        "_pending_status_reports": status_reports,
        "step": step,
        "inventory": inventory,
        "event_log": log,
    }


def node_monitor(state: OrchestrationState) -> dict:
    reports: list[StatusReport] = state.get("_pending_status_reports", [])
    task_graph = state["task_graph"]
    registry = state["registry"]
    inventory = dict(state.get("inventory", {}))
    log = state.get("event_log", [])

    trigger = None
    for report in reports:
        t = process_status(report, task_graph, registry, inventory)
        if t:
            trigger = t
            log.append(f"[monitor] replan triggered: {t.task_id} — {t.reason}")

    return {
        "task_graph": task_graph,
        "registry": registry,
        "inventory": inventory,
        "last_trigger": trigger,
        "_pending_status_reports": [],
        "event_log": log,
    }


def node_check_done(state: OrchestrationState) -> dict:
    task_graph = state["task_graph"]
    all_done = all(t.status == "done" for t in task_graph.tasks)
    step = state.get("step", 0)
    log = state.get("event_log", [])
    if all_done:
        log.append(f"[done] All tasks complete in {step} steps.")
    return {"done": all_done, "event_log": log}


def node_replan(state: OrchestrationState) -> dict:
    trigger = state["last_trigger"]
    log = state.get("event_log", [])
    log.append(f"[replan] handling failure of '{trigger.task_id}': {trigger.reason}")
    return {"event_log": log}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_validate(state: OrchestrationState) -> Literal["dispatch", "coordinate", "check_done"]:
    retries = state.get("validate_retries", 0)
    batch = state.get("pending_assignments")
    if batch and batch.assignments:
        return "dispatch"
    if retries >= _MAX_VALIDATE_RETRIES:
        return "check_done"  # give up on this round
    return "coordinate"


def route_after_check_done(state: OrchestrationState) -> Literal["coordinate", "replan", "__end__"]:
    if state.get("done"):
        return END
    if state.get("step", 0) >= _MAX_STEPS:
        return END
    if state.get("last_trigger"):
        return "replan"
    return "coordinate"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(env: EnvInterface) -> Any:
    """Return a compiled LangGraph runnable bound to the given env."""

    def _dispatch(state: OrchestrationState) -> dict:
        return node_dispatch(state, env)

    g = StateGraph(OrchestrationState)
    g.add_node("init", node_init)
    g.add_node("coordinate", node_coordinate)
    g.add_node("validate", node_validate)
    g.add_node("dispatch", _dispatch)
    g.add_node("monitor", node_monitor)
    g.add_node("check_done", node_check_done)
    g.add_node("replan", node_replan)

    g.set_entry_point("init")
    g.add_edge("init", "coordinate")
    g.add_edge("coordinate", "validate")
    g.add_conditional_edges("validate", route_after_validate)
    g.add_edge("dispatch", "monitor")
    g.add_edge("monitor", "check_done")
    g.add_conditional_edges("check_done", route_after_check_done)
    g.add_edge("replan", "coordinate")

    return g.compile()


def run_orchestration(
    blueprint_id: str,
    env: EnvInterface,
    registry: RobotRegistry,
    initial_resources: dict[str, int] | None = None,
    coordinator_mode: Literal["greedy", "llm"] = "llm",
    seed: int = 42,
) -> OrchestrationState:
    """Convenience runner. Returns final state."""
    obs = env.reset(blueprint_id=blueprint_id, seed=seed)

    site = SiteData(
        terrain=obs.terrain,
        buildable_mask=[[True] * len(obs.terrain[0]) for _ in obs.terrain],
        hazard_zones=obs.hazards,
        resource_nodes=[],
    )

    initial: OrchestrationState = {
        "blueprint_id": blueprint_id,
        "blueprint": None,
        "site": site,
        "task_graph": None,
        "registry": registry,
        "inventory": initial_resources or dict(obs.resources),
        "pending_assignments": None,
        "last_trigger": None,
        "last_observation": obs,
        "step": 0,
        "done": False,
        "coordinator_mode": coordinator_mode,
        "validate_retries": 0,
        "event_log": [],
        "_pending_status_reports": [],
    }

    graph = build_graph(env)
    final = graph.invoke(initial, config={"recursion_limit": 500})
    return final
