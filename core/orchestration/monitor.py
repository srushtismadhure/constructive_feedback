"""Ingests StatusReports from the sim; updates task/robot state; emits replan triggers."""

from __future__ import annotations

from orchestration.contracts import ReplanTrigger, RobotRegistry, StatusReport, TaskGraph


def process_status(
    report: StatusReport,
    task_graph: TaskGraph,
    registry: RobotRegistry,
    inventory: dict[str, int],
) -> ReplanTrigger | None:
    """Update in-place state and return a ReplanTrigger if replanning is needed."""
    task = next((t for t in task_graph.tasks if t.task_id == report.task_id), None)
    robot = registry.get(report.robot_id)

    if task is None or robot is None:
        print(f"[monitor] unknown task/robot in report: {report}")
        return None

    if report.result == "complete":
        task.status = "done"
        robot.status = "idle"
        robot.position = report.new_position
        for material, qty in report.resources_consumed.items():
            inventory[material] = max(0, inventory.get(material, 0) - qty)
        return None

    # failed or blocked
    task.status = "failed"
    robot.status = "idle"
    robot.position = report.new_position

    available_robots = [r.id for r in registry.idle_robots()]
    return ReplanTrigger(
        task_id=report.task_id,
        reason=report.reason or report.result,
        available_robots=available_robots,
    )
