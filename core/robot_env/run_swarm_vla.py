"""Run a VLA policy through the Mars swarm pick-and-place task; report a success rate.

The HUD SDK owns the loop and the `robot` (openpi/0) wire protocol; each rollout is one
seeded task (the scene is deterministic, so seeds vary nothing today — `--group` just
repeats the task), and reward is the bridge's shaped placement score.

    # plumbing check (no GPU, no model):
    uv run python robot_env/run_swarm_vla.py --noop

    # a remote pi0.5 checkpoint served on a GPU box / Modal (serve/policy_server_mars.py):
    uv run python robot_env/run_swarm_vla.py --remote HOST:PORT --group 4
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from hud import LocalRuntime, Taskset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from swarm_hud_env import swarm_pick  # noqa: E402


async def run_eval(args: argparse.Namespace) -> None:
    if args.record:
        # The SDK's RobotAgent records a LeRobot dataset when its recorder dir is set.
        os.environ["HUD_ROBOT_RECORD_DIR"] = args.record

    if args.noop:
        from swarm_agents import NoopAgent
        agent, policy = NoopAgent(), "noop"
    elif args.remote:
        from swarm_agents import RemoteAgent
        host, _, port = args.remote.rpartition(":")
        agent = RemoteAgent(host=host or "localhost", port=int(port))
        policy = f"remote://{args.remote}"
    else:
        raise SystemExit("pass --noop or --remote HOST:PORT")

    if args.record:
        agent.save = True  # SDK Recorder writes (obs, action, task) frames per rollout

    tasks = [swarm_pick(seed=i) for i in range(args.group)]
    print(f"swarm VLA eval: {args.group} rollout(s) (policy: {policy})\n")

    job = await Taskset("mars-swarm-vla", tasks).run(
        agent, runtime=LocalRuntime(str(ROOT / "swarm_hud_env.py")),
        max_concurrent=args.max_concurrent,
    )

    rewards = [run.reward or 0.0 for run in job.runs]
    successes = [r >= args.threshold for r in rewards]
    for i, (r, ok) in enumerate(zip(rewards, successes, strict=False)):
        print(f"  rollout {i:>2}: reward={r:.4f}  {'SUCCESS' if ok else 'fail'}")
    n = len(rewards) or 1
    print(f"\n{'=' * 50}\nSUCCESS RATE: {sum(successes) / n * 100:.1f}%   "
          f"mean reward: {sum(rewards) / n:.4f}\n{'=' * 50}")
    if args.record:
        print(f"dataset recorded under: {args.record}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a VLA on the Mars swarm pick task.")
    ap.add_argument("--noop", action="store_true", help="no-op agent (no GPU/model) — wire check")
    ap.add_argument("--remote", default=None, metavar="HOST:PORT", help="remote openpi/0 server")
    ap.add_argument("--group", type=int, default=4, metavar="N", help="rollouts (seeds 0..N-1)")
    ap.add_argument("--threshold", type=float, default=0.999, metavar="R", help="reward>=R success")
    ap.add_argument("--max-concurrent", type=int, default=1, metavar="N", help="parallel rollouts")
    ap.add_argument("--record", default=None, metavar="DIR", help="record to a LeRobot dataset dir")
    asyncio.run(run_eval(ap.parse_args()))


if __name__ == "__main__":
    main()
