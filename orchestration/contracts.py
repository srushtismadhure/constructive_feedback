"""All shared Pydantic schemas. Define here first; every module imports from here."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Blueprint (raw input)
# ---------------------------------------------------------------------------

class Component(BaseModel):
    id: str
    type: Literal["foundation", "wall", "roof", "panel"]
    position: tuple[int, int, int]
    material: str
    quantity: int
    depends_on: list[str] = []


class Blueprint(BaseModel):
    building_id: str
    components: list[Component]


# ---------------------------------------------------------------------------
# Parsed blueprint (parser output)
# ---------------------------------------------------------------------------

class ParsedComponent(Component):
    required_role: Literal["excavator", "welder", "hauler"]


class ParsedBlueprint(BaseModel):
    building_id: str
    components: list[ParsedComponent]
    dependency_order: list[str]  # topologically sorted component ids


# ---------------------------------------------------------------------------
# Site data (site analysis output)
# ---------------------------------------------------------------------------

class SiteData(BaseModel):
    terrain: list[list[float]]
    buildable_mask: list[list[bool]]
    hazard_zones: list[tuple[int, int]] = []
    resource_nodes: list[tuple[int, int]] = []


# ---------------------------------------------------------------------------
# Task and task graph (sequencer output)
# ---------------------------------------------------------------------------

class Task(BaseModel):
    task_id: str
    action: Literal["excavate", "haul", "place", "weld"]
    required_role: Literal["excavator", "welder", "hauler"]
    position: tuple[int, int, int]
    material: str | None = None
    quantity: int | None = None
    depends_on: list[str] = []
    status: Literal["pending", "assigned", "in_progress", "done", "failed"] = "pending"
    component_id: str = ""


class TaskGraph(BaseModel):
    building_id: str
    tasks: list[Task]


# ---------------------------------------------------------------------------
# Robot and registry
# ---------------------------------------------------------------------------

class Robot(BaseModel):
    id: str
    role: Literal["excavator", "welder", "hauler"]
    capabilities: list[str]
    status: Literal["idle", "busy", "charging", "broken"] = "idle"
    position: tuple[int, int]
    carrying: dict[str, int] = {}


class RobotRegistry(BaseModel):
    robots: list[Robot]

    def get(self, robot_id: str) -> Robot | None:
        return next((r for r in self.robots if r.id == robot_id), None)

    def idle_robots(self, role: Literal["excavator", "welder", "hauler"] | None = None) -> list[Robot]:
        return [
            r for r in self.robots
            if r.status == "idle" and (role is None or r.role == role)
        ]


# ---------------------------------------------------------------------------
# Assignment (coordinator output)
# ---------------------------------------------------------------------------

class Assignment(BaseModel):
    robot_id: str
    task_id: str


class AssignmentBatch(BaseModel):
    assignments: list[Assignment]
    reasoning: str | None = None
    llm_fallback: bool = False  # True when LLM failed and greedy was used instead


# ---------------------------------------------------------------------------
# Status report (sim → monitor)
# ---------------------------------------------------------------------------

class StatusReport(BaseModel):
    robot_id: str
    task_id: str
    result: Literal["complete", "failed", "blocked"]
    reason: str | None = None
    new_position: tuple[int, int]
    resources_consumed: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Environment interface types
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    terrain: list[list[float]]
    placed: list[list[int]]
    robots: list[Robot]
    resources: dict[str, int]
    hazards: list[tuple[int, int]]
    step: int


class Action(BaseModel):
    robot_id: str
    command: Literal["move", "excavate", "pickup", "place", "weld", "noop"]
    target: tuple[int, int, int] | None = None
    material: str | None = None


# ---------------------------------------------------------------------------
# Internal orchestration types
# ---------------------------------------------------------------------------

class ReplanTrigger(BaseModel):
    task_id: str
    reason: str
    available_robots: list[str] = Field(default_factory=list)


class OrchestrationState(TypedDict):
    blueprint_id: str
    blueprint: ParsedBlueprint | None
    site: SiteData | None
    task_graph: TaskGraph | None
    registry: RobotRegistry | None
    inventory: dict[str, int]
    pending_assignments: AssignmentBatch | None
    last_trigger: ReplanTrigger | None
    last_observation: Observation | None
    step: int
    done: bool
    coordinator_mode: Literal["greedy", "llm"]
    validate_retries: int
    event_log: list[str]
    _pending_status_reports: list[StatusReport]
