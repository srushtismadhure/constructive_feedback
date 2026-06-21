"""End-to-end smoke test: bridge reward → rover controller → Brain drives the rover.

Run from the repo root with an interpreter that has `mujoco` + `langgraph`:

    robot_training/.venv/bin/python tests/smoke_test.py

Exits non-zero on the first failed check. No HUD key or LLM key needed
(uses the greedy coordinator).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestration.contracts import Action, Robot, RobotRegistry
from orchestration.graph import run_orchestration
from orchestration.mujoco_adapter import RoverEnvAdapter

DEFAULT_FLEET = RobotRegistry(robots=[
    Robot(id="excavator-1", role="excavator", capabilities=["excavate"], position=(1, 1)),
    Robot(id="excavator-2", role="excavator", capabilities=["excavate"], position=(6, 6)),
    Robot(id="hauler-1", role="hauler", capabilities=["haul", "pickup"], position=(2, 2)),
    Robot(id="hauler-2", role="hauler", capabilities=["haul", "pickup"], position=(5, 5)),
    Robot(id="welder-1", role="welder", capabilities=["weld", "place"], position=(3, 1)),
    Robot(id="welder-2", role="welder", capabilities=["weld", "place"], position=(4, 6)),
])


def test_bridge_reward() -> None:
    """Driving toward the goal produces positive progress reward."""
    env = RoverEnvAdapter(render=False)
    env.reset("habitat-dome", seed=0)
    _, reward, _, info = env.step(Action(robot_id="r", command="place", target=(0, 1, 0)))
    assert info.get("rejection_reason") is None, info["rejection_reason"]
    assert reward > 0, f"expected positive progress reward, got {reward}"
    env.close()
    print(f"  bridge reward: reached target, reward={reward:+.2f}  OK")


def test_controller_reaches_targets() -> None:
    """The pure-pursuit controller reaches grid targets across the plane."""
    env = RoverEnvAdapter(render=False)
    env.reset("habitat-dome", seed=0)
    targets = [(2, 3, 0), (5, 5, 0), (0, 1, 0), (6, 2, 0), (3, 7, 0), (7, 0, 0)]
    reached = 0
    for tgt in targets:
        _, _, _, info = env.step(Action(robot_id="r", command="place", target=tgt))
        reached += info.get("rejection_reason") is None
    env.close()
    assert reached == len(targets), f"only {reached}/{len(targets)} targets reached"
    print(f"  controller: {reached}/{len(targets)} targets reached  OK")


def test_brain_drives_rover() -> None:
    """The Brain plans habitat-dome and the rover executes every task to done."""
    env = RoverEnvAdapter(render=False)
    final = run_orchestration(
        "habitat-dome", env=env, registry=DEFAULT_FLEET, coordinator_mode="greedy"
    )
    env.close()
    assert final["done"], f"build did not complete (steps={final['step']})"
    print(f"  brain e2e: build complete in {final['step']} steps  OK")


def main() -> int:
    checks = [test_bridge_reward, test_controller_reaches_targets, test_brain_drives_rover]
    for check in checks:
        print(f"[{check.__name__}]")
        try:
            check()
        except AssertionError as exc:
            print(f"  FAIL: {exc}")
            return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
