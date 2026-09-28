"""Render control episodes to GIF: find the best seed, then animate it."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch

from training.control.evaluate import load_policy
from training.control.obs_adapter import ObsWindow
from training.control.rollout import run_episode


def greedy_action(policy, window_tensor: torch.Tensor) -> int:
    """Deterministic argmax action — the deployed policy, and reproducible."""
    with torch.no_grad():
        q = policy.action_head(policy.q_features(window_tensor))
    return int(q.argmax(dim=-1).item())


def scan_seeds(checkpoint: str, arch: str, seed_base: int, n: int) -> list[tuple[int, float]]:
    """Evaluate fixed seeds greedily; return (seed, return) sorted best-first."""
    policy = load_policy(arch, checkpoint)
    env = gym.make("LunarLander-v3")
    results = []
    for i in range(n):
        obs, _ = env.reset(seed=seed_base + i)
        window = ObsWindow(policy.window, policy.adapter.obs_dim)
        window.push(torch.as_tensor(obs, dtype=torch.float32))
        total, done = 0.0, False
        while not done:
            action = greedy_action(policy, window.tensor())
            for _ in range(3):  # frame-skip, matching training
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
                if done:
                    break
            window.push(torch.as_tensor(obs, dtype=torch.float32))
        results.append((seed_base + i, total))
        print(f"seed {seed_base + i} | return {total:+8.1f}")
    env.close()
    return sorted(results, key=lambda r: r[1], reverse=True)


def render_episode(checkpoint: str, arch: str, seed: int, out_path: str,
                    action_repeat: int = 3) -> float:
    """Render one deterministic episode to GIF; returns its return."""
    policy = load_policy(arch, checkpoint)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env.action_space.seed(seed)

    obs, _ = env.reset(seed=seed)
    window = ObsWindow(policy.window, policy.adapter.obs_dim)
    window.push(torch.as_tensor(obs, dtype=torch.float32))

    frames = []
    total = 0.0
    done = False
    while not done:
        frames.append(env.render())  # one frame per decision (frame-skip aware)
        action = greedy_action(policy, window.tensor())
        for _ in range(action_repeat):
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
            if done:
                break
        window.push(torch.as_tensor(obs, dtype=torch.float32))

    from PIL import Image

    imgs = [Image.fromarray(frame) for frame in frames]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        out, save_all=True, append_images=imgs[1:], duration=50, loop=0
    )
    env.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the best demo episode")
    parser.add_argument("--arch", default="configs/arch_30m.yaml")
    parser.add_argument("--checkpoint", default="runs/p6/control_dqn_v4.pt")
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--scan", type=int, default=30)
    parser.add_argument("--out", default="docs/assets/landing.gif")
    args = parser.parse_args()

    print(f"scanning {args.scan} seeds from {args.seed_base}...")
    ranked = scan_seeds(args.checkpoint, args.arch, args.seed_base, args.scan)
    best_seed, best_return = ranked[0]
    print(f"\nbest seed: {best_seed} (return {best_return:+.1f})")

    total = render_episode(args.checkpoint, args.arch, best_seed, args.out)
    print(f"rendered {args.out} | episode return {total:+.1f}")


if __name__ == "__main__":
    main()
