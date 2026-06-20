import argparse
import asyncio

from run_hud_demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Mars MuJoCo viewer.")
    parser.add_argument(
        "--autopilot",
        action="store_true",
        help="Drive the scene through the HUD bridge so something visibly moves.",
    )
    parser.add_argument("--rate-hz", type=float, default=10.0)
    args = parser.parse_args()

    if not args.autopilot:
        print("Nothing is animated in the passive scene. Use --autopilot to drive HUD actions.")
        print("Example: mjpython robot_env/simulate_rover.py --autopilot")
        return

    asyncio.run(run_demo(args.rate_hz))


if __name__ == "__main__":
    main()
