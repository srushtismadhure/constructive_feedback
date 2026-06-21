"""ParsedBlueprint → TaskGraph.

Each component expands into 1-2 tasks:
  - foundation  → excavate (excavator)
  - wall/roof   → haul (hauler) then weld (welder); haul must precede weld
  - panel       → haul (hauler) then place (welder); haul must precede place
"""

from __future__ import annotations

from orchestration.contracts import ParsedBlueprint, Task, TaskGraph


def sequence(blueprint: ParsedBlueprint) -> TaskGraph:
    tasks: list[Task] = []
    # component_id → list of task_ids generated for it
    comp_tasks: dict[str, list[str]] = {}

    for comp_id in blueprint.dependency_order:
        comp = next(c for c in blueprint.components if c.id == comp_id)

        # Resolve prerequisite task_ids: for each dependency component, we need
        # its *last* task to be done before we start.
        dep_task_ids: list[str] = []
        for dep_comp_id in comp.depends_on:
            dep_tasks = comp_tasks.get(dep_comp_id, [])
            if dep_tasks:
                dep_task_ids.append(dep_tasks[-1])

        if comp.type == "foundation":
            t = Task(
                task_id=f"{comp_id}_excavate",
                action="excavate",
                required_role="excavator",
                position=comp.position,
                material=comp.material,
                quantity=comp.quantity,
                depends_on=dep_task_ids,
                component_id=comp_id,
            )
            tasks.append(t)
            comp_tasks[comp_id] = [t.task_id]

        elif comp.type in ("extrude",):
            t = Task(
                task_id=f"{comp_id}_extrude",
                action="extrude",
                required_role="3d_printer",
                position=comp.position,
                material=comp.material,
                quantity=comp.quantity or 1,
                depends_on=dep_task_ids,
                component_id=comp_id,
                params=comp.params,
            )
            tasks.append(t)
            comp_tasks[comp_id] = [t.task_id]

        elif comp.type in ("place",):
            t = Task(
                task_id=f"{comp_id}_place",
                action="place",
                required_role="arm_robot",
                position=comp.position,
                material=comp.material,
                quantity=comp.quantity or 1,
                depends_on=dep_task_ids,
                component_id=comp_id,
                params=comp.params,
            )
            tasks.append(t)
            comp_tasks[comp_id] = [t.task_id]

        elif comp.type in ("wall", "roof", "panel"):
            haul_id = f"{comp_id}_haul"
            action = "weld" if comp.type in ("wall", "roof") else "place"
            place_id = f"{comp_id}_{action}"

            haul = Task(
                task_id=haul_id,
                action="haul",
                required_role="hauler",
                position=comp.position,
                material=comp.material,
                quantity=comp.quantity,
                depends_on=dep_task_ids,
                component_id=comp_id,
            )
            place = Task(
                task_id=place_id,
                action=action,
                required_role="welder",
                position=comp.position,
                material=comp.material,
                quantity=comp.quantity,
                depends_on=[haul_id],
                component_id=comp_id,
            )
            tasks.extend([haul, place])
            comp_tasks[comp_id] = [haul_id, place_id]

    return TaskGraph(building_id=blueprint.building_id, tasks=tasks)
