"""P6.U4: ablation suite + per-phase routing evidence on the best checkpoint."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="P6.U4 evidence run")
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--checkpoint", default="runs/p6/control_dqn_v3.pt")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    from training.control.ablate import ablation_suite
    from training.control.evaluate import load_policy, phase_telemetry

    print("== routing ablations (same trunk, forced routing) ==")
    results = ablation_suite(args.arch, args.checkpoint, episodes=args.episodes)
    for name, r in results.items():
        print(f"{name:<10} return {r.mean_return:+8.1f} +/- {r.std_return:6.1f} "
              f"| natural terminations {r.landings}/{r.episodes}")

    print("\n== per-phase routing (learned policy) ==")
    policy = load_policy(args.arch, args.checkpoint)
    telemetry = phase_telemetry(policy, episodes=5)
    for phase, stats in telemetry.items():
        print(f"{phase:<6} mean P(ROUTE) {stats['mean_route_frac']:.3f} "
              f"({stats['steps']} steps)")


if __name__ == "__main__":
    main()
