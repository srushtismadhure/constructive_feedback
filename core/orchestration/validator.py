"""Pre-dispatch sanity checks. Runs before any action hits the sim."""

from __future__ import annotations

from orchestration.contracts import AssignmentBatch, RobotRegistry, Task, TaskGraph


def validate_assignments(
    batch: AssignmentBatch,
    task_graph: TaskGraph,
    registry: RobotRegistry,
    inventory: dict[str, int],
) -> list[str]:
    """Return list of rejection reasons. Empty list means all assignments pass."""
    errors: list[str] = []
    task_map = {t.task_id: t for t in task_graph.tasks}
    done_ids = {t.task_id for t in task_graph.tasks if t.status == "done"}
    assigned_task_ids: set[str] = set()

    for assignment in batch.assignments:
        robot = registry.get(assignment.robot_id)
        task = task_map.get(assignment.task_id)

        if robot is None:
            errors.append(f"Unknown robot '{assignment.robot_id}'")
            continue
        if task is None:
            errors.append(f"Unknown task '{assignment.task_id}'")
            continue

        # Role match
        if robot.role != task.required_role:
            errors.append(
                f"Role mismatch: robot '{robot.id}' is {robot.role}, "
                f"task '{task.task_id}' needs {task.required_role}"
            )

        # Robot available
        if robot.status != "idle":
            errors.append(f"Robot '{robot.id}' is {robot.status}, not idle")

        # Dependencies satisfied
        unsatisfied = [dep for dep in task.depends_on if dep not in done_ids]
        if unsatisfied:
            errors.append(
                f"Task '{task.task_id}' has unsatisfied dependencies: {unsatisfied}"
            )

        # No double-assignment to same task
        if assignment.task_id in assigned_task_ids:
            errors.append(f"Task '{task.task_id}' assigned to multiple robots")
        assigned_task_ids.add(assignment.task_id)

        # Resource check
        if task.material and task.quantity:
            available = inventory.get(task.material, 0)
            if available < task.quantity:
                errors.append(
                    f"Insufficient {task.material}: need {task.quantity}, have {available}"
                )

    return errors
