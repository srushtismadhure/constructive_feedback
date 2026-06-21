"""HUD training run for the Mars 3-D printer arm.

Runs the scripted policy (guaranteed 20/20) as the data-collection
rollout and records each episode to HUD as a trace.  With --runs N
the N rewards are turned into GRPO advantages ready for an optimizer.

Usage:
    python train_printer.py                   # 1 scripted rollout, dome
    python train_printer.py --runs 4          # 4 rollouts, compute GRPO advantages
    python train_printer.py --structure dome --runs 4 --random-policy
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Ensure robot_env/ is resolvable regardless of CWD.
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# Load core/.env (one level up from robot_env/) so HUD_API_KEY etc. are
# available without a manual `export` in the shell.
from dotenv import load_dotenv
load_dotenv(os.path.join(_DIR, "..", ".env"))

import numpy as np

from hud_printer_bridge import MarsPrinterBridge, PRINT_STRUCTURES, _scripted_printer_actions


async def _run_episode(structure_id: str, seed: int, use_scripted: bool) -> dict:
    """Run one full episode and return the result dict."""
    bridge = MarsPrinterBridge(render=False)
    prompt = await bridge.reset(task_id=structure_id, seed=seed)

    if use_scripted:
        actions = _scripted_printer_actions(structure_id)
    else:
        # Random policy baseline — rarely succeeds but useful for GRPO contrast.
        rng = np.random.default_rng(seed)
        actions = [rng.uniform(-1, 1, size=5).astype(np.float32) for _ in range(3000)]

    t0 = time.time()
    for action in actions:
        bridge.step(action)
        _, terminated = bridge.get_observation()
        if terminated:
            break

    elapsed = time.time() - t0
    result = bridge.result()
    result["seed"] = seed
    result["elapsed_s"] = round(elapsed, 2)
    result["prompt"] = prompt
    bridge.close()
    return result


async def run_training(structure_id: str, n_runs: int, use_scripted: bool,
                       mixed: bool = False) -> None:
    n_waypoints = len(PRINT_STRUCTURES[structure_id]())
    policy = "mixed (scripted+random)" if mixed else ("scripted" if use_scripted else "random")
    print(f"[train] structure={structure_id}  waypoints={n_waypoints}  runs={n_runs}  policy={policy}")

    # Run all episodes (sequential — MuJoCo is not thread-safe).
    # In mixed mode, even seeds run scripted and odd seeds run random so the
    # group has reward variance (non-zero GRPO advantages).
    results = []
    for seed in range(n_runs):
        episode_scripted = use_scripted if not mixed else (seed % 2 == 0)
        tag = "scripted" if episode_scripted else "random"
        print(f"  episode {seed + 1}/{n_runs}  seed={seed}  [{tag}] ...", end=" ", flush=True)
        result = await _run_episode(structure_id, seed, episode_scripted)
        result["policy"] = tag
        results.append(result)
        print(f"score={result['score']:.3f}  printed={result['printed_count']}/{result['total_waypoints']}  "
              f"success={result['success']}  t={result['elapsed_s']}s")

    # Summary
    scores = [r["score"] for r in results]
    mean_score = sum(scores) / len(scores)
    print(f"\n=== Summary ===")
    print(f"  mean score : {mean_score:.3f}")
    print(f"  successes  : {sum(r['success'] for r in results)}/{n_runs}")

    # GRPO advantages (useful when mixing scripted + random for contrastive training).
    if n_runs >= 2:
        mean_r = sum(scores) / len(scores)
        std_r = (sum((s - mean_r) ** 2 for s in scores) / len(scores)) ** 0.5 or 1.0
        advantages = [(s - mean_r) / std_r for s in scores]
        print(f"\n  GRPO advantages (group_relative, normalize_std=True):")
        for i, (r, adv) in enumerate(zip(results, advantages)):
            print(f"    seed={r['seed']}  score={r['score']:.3f}  adv={adv:+.4f}")

    # Record to HUD if API key is present.
    hud_key = os.environ.get("HUD_API_KEY", "")
    if hud_key:
        try:
            from hud import Trace
            from hud.types import Step
            for r in results:
                trace = Trace()
                trace.record(Step(
                    source="system",
                    extra={
                        "structure": structure_id,
                        "seed": r["seed"],
                        "printed": r["printed_count"],
                        "total": r["total_waypoints"],
                        "success": r["success"],
                        "policy": r.get("policy", "scripted" if use_scripted else "random"),
                        "score": r["score"],
                    },
                ))
                print(f"  [hud] recorded trace  score={r['score']:.3f}  steps={len(trace.steps)}")
        except Exception as exc:
            print(f"  [hud] recording skipped: {exc}")
    else:
        print("\n  [hud] HUD_API_KEY not set — runs not recorded to platform.")


async def run_all(n_runs: int, use_scripted: bool, mixed: bool) -> None:
    """Sweep every registered structure — covers all test scenarios."""
    structures = list(PRINT_STRUCTURES)
    print(f"==== Training sweep over {len(structures)} scenario(s): {structures} ====\n")
    for i, structure_id in enumerate(structures):
        print(f"---- scenario {i + 1}/{len(structures)}: {structure_id} ----")
        await run_training(structure_id, n_runs, use_scripted, mixed)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="HUD training run for Mars 3-D printer arm.")
    parser.add_argument("--structure", default="dome", choices=list(PRINT_STRUCTURES))
    parser.add_argument("--runs", type=int, default=1, help="Number of rollouts.")
    parser.add_argument("--random-policy", action="store_true",
                        help="Use random actions instead of scripted policy.")
    parser.add_argument("--all", action="store_true",
                        help="Sweep every registered structure (all test scenarios).")
    parser.add_argument("--mixed", action="store_true",
                        help="Mix scripted (even seeds) + random (odd seeds) for non-zero GRPO advantages.")
    args = parser.parse_args()

    if args.all:
        asyncio.run(run_all(args.runs, not args.random_policy, args.mixed))
    else:
        asyncio.run(run_training(args.structure, args.runs, not args.random_policy, args.mixed))


if __name__ == "__main__":
    main()
