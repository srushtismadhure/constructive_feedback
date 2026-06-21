"""Blueprint JSON → ParsedBlueprint with role assignment and topological order."""

from __future__ import annotations

import json
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Literal

from orchestration.contracts import Blueprint, ParsedBlueprint, ParsedComponent

_BLUEPRINTS_DIR = Path(__file__).parent / "blueprints"

_ROLE_MAP: dict[str, Literal["excavator", "welder", "hauler"]] = {
    "foundation": "excavator",
    "wall": "welder",
    "roof": "welder",
    "panel": "welder",
}


def _topo_sort(components: list[ParsedComponent]) -> list[str]:
    graph: dict[str, set[str]] = {c.id: set(c.depends_on) for c in components}
    sorter = TopologicalSorter(graph)
    return list(sorter.static_order())


def parse_blueprint(source: str | dict) -> ParsedBlueprint:
    """Parse a blueprint from a file path, blueprint_id string, or dict.

    Args:
        source: blueprint_id (e.g. "habitat-dome"), absolute file path, or raw dict.
    """
    if isinstance(source, dict):
        raw = Blueprint.model_validate(source)
    elif isinstance(source, str) and source.endswith(".json"):
        raw = Blueprint.model_validate_json(Path(source).read_text())
    else:
        # Treat as blueprint_id — look up in bundled blueprints dir
        slug = source.replace("-", "_")
        candidate = _BLUEPRINTS_DIR / f"{slug}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"No bundled blueprint for id '{source}' (tried {candidate})")
        raw = Blueprint.model_validate_json(candidate.read_text())

    parsed_components = [
        ParsedComponent(**c.model_dump(), required_role=_ROLE_MAP[c.type])
        for c in raw.components
    ]

    return ParsedBlueprint(
        building_id=raw.building_id,
        components=parsed_components,
        dependency_order=_topo_sort(parsed_components),
    )


def load_bundled_blueprints() -> dict[str, ParsedBlueprint]:
    """Return all bundled blueprints keyed by building_id."""
    result: dict[str, ParsedBlueprint] = {}
    for path in _BLUEPRINTS_DIR.glob("*.json"):
        try:
            bp = parse_blueprint(str(path))
            result[bp.building_id] = bp
        except Exception as exc:
            print(f"[parser] skipping {path.name}: {exc}")
    return result
