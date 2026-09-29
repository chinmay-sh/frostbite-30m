# Frostbite-30M · v7 — Consolidation run on Kaggle (overnight, browser closed) — EXECUTED, see D34

> **Outcome (D34, 2026-09-29):** run completed, 1500/1500 episodes. Deployed-policy
> (A-mode) median −200.7 → **−127.6** (+73), worst −321.6 → −214.9, 8/25 > −100;
> B-mode mean −224.0 → −118.9; A↔C gap inverted (argmax now beats stochastic routing).
> Verdict + full matrix in `docs/DECISIONS.md` D34. This document is the reproducible
> run protocol — kept as the recipe for any future continuation.

Continue Double-DQN training from `control_dqn_v6.pt` with **ε pinned at 0.02** (no re-exploration),
D32 fixes active (Polyak soft targets τ=0.005 + PER α=0.6), shaping ON for training, evaluation always raw.

| knob | value | why |
| --- | --- | --- |
| resume | v6 checkpoint | best stable ckpt (median −98.3, crash tail gone) |
| ε | 0.02 fixed | consolidation regime — pure exploitation of the trained Q |
| episodes | 1500 | ≈4–6 h, fits Kaggle's 12 h GPU session cap |
| warmup | 0 decisions | resuming a trained Q, not starting fresh |
| shaping | ON (train), raw eval | same protocol as v4/v5/v6 |
| trunk_lr | off (head only) | D24 — substrates & routing stay banked |

The whole training path below was smoke-tested locally (3 episodes from v6: flat ε verified,
checkpoint/buffer saves round-trip). Cell 1 is the Kaggle-verified install order:
**model deps (`ncps`, `pyyaml`) install separately from the env stack** — box2d-py builds
from source on Kaggle and needs `apt-get install swig`; if it fails inside a shared pip
command, `ncps` never lands and cell 3 dies on `import ncps`.

---

## One-time Kaggle setup (~5 min)

1. **Upload the checkpoint as a private Dataset**: kaggle.com → *Datasets* → *New Dataset* →
   upload `runs/p6/control_dqn_v6.pt` (~118 MB) → name it `frostbite-v6` → Private → Create.
2. **New Notebook** → *Add Input* → *Your Datasets* → `frostbite-v6`.
   It will appear at `/kaggle/input/frostbite-v6/control_dqn_v6.pt`.
3. **Settings** (right panel):
   - *Accelerator* → **GPU T4 x2** (or P100 — the model is tiny; anything works)
   - *Internet* → **ON** (needed for `git clone` + `pip`; requires a phone-verified account.
     If you can't enable it: upload the repo as a zip dataset too and unzip instead of cloning.)
4. **Quota reality**: GPU sessions cap at 12 h, quota ~30 h/week. A 4–6 h run fits easily.

## Two overnight modes

- **Interactive + close browser** (simplest): run the cells, watch cell 6 start training, close the
  tab. The session keeps running server-side; come back, reconnect, artifacts are in `/kaggle/working`.
- **Save & Run All (Commit)** (most robust): *Save Version → Save & Run All (Commit)*. The whole
  notebook runs headless; when it finishes, `/kaggle/working` becomes the version's **Output** —
  downloadable even days later. This is the fire-and-forget option.

In both modes the autosave in cell 6 writes `control_dqn_v7.pt` + `v7_buffer.pt` to
`/kaggle/working` every 30 min, so a crashed interactive session usually still has a recent
checkpoint in the file browser (worst case: lose 30 min).

---

## Cell 1 — clone repo + deps (idempotent: safe to re-run after a kernel restart)

```python
%cd /kaggle/working
import os
if not os.path.isdir("frostbite-30m/.git"):
    !git clone --depth 1 https://github.com/chinmay-sh/frostbite-30m.git
%cd frostbite-30m

# Model deps first — a failure here blocks everything else, so keep them
# separate from the env deps below. (No-op if already installed.)
!pip -q install ncps pyyaml

# box2d-py builds from source on Kaggle (no wheel) and needs swig; Kaggle's
# preinstalled gymnasium is 1.2.0 — the repo wants >=1.3.0.
!apt-get -qq install -y swig
!pip -q install "gymnasium[box2d]>=1.3.0"

import torch, gymnasium
from importlib.metadata import version
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| gymnasium", gymnasium.__version__, "| ncps", version("ncps"))
assert tuple(int(x) for x in gymnasium.__version__.split(".")[:2]) >= (1, 3), \
    f"stale gymnasium {gymnasium.__version__} — restart the kernel and re-run this cell"
```

> Two Kaggle traps live here:
> 1. If both `pip` lines share one command and box2d fails, pip aborts the whole
>    install and `ncps` silently never lands (`ModuleNotFoundError: ncps`).
> 2. Upgrading gymnasium mid-session does NOT change what an already-running
>    kernel reports — Python caches the imported module. If the assert fires,
>    the pip install worked but the kernel is stale: **Run → Restart kernel**
>    (keeps the disk: clone + installs survive) and re-run this cell.
> The `kaggle-environments`/`dopamine-rl` resolver warnings are cosmetic.

## Cell 2 — stage the v6 checkpoint from the attached dataset

```python
import glob, os, shutil

os.makedirs("runs/p6", exist_ok=True)
hits = glob.glob("/kaggle/input/**/control_dqn_v6.pt", recursive=True)
assert hits, "attach the frostbite-v6 dataset (Add Input) and retry"
shutil.copy2(hits[0], "runs/p6/control_dqn_v6.pt")

import torch
p = torch.load("runs/p6/control_dqn_v6.pt", map_location="cpu", weights_only=True)
print("checkpoint keys:", list(p.keys()))
h = p.get("history", [])
print(f"v6 history: {len(h)} episodes | last-5 returns:",
      [f"{e['return']:+.0f}" for e in h[-5:]])
```

## Cell 3 — build model, load checkpoint, sanity check

```python
import sys, torch
sys.path.insert(0, ".")

from frostbite.config.arch import ArchConfig
from frostbite.model import FrostbiteModel
from frostbite.utils import resolve_device, count_params, seed_everything
from training.control.policy_wrapper import ControlPolicy

seed_everything(7)
device = resolve_device()
arch = ArchConfig.from_yaml("configs/arch_30m.yaml")
model = FrostbiteModel(arch).to(device)
payload = torch.load("runs/p6/control_dqn_v6.pt", map_location=device, weights_only=True)
model.load_state_dict(payload["model"])
policy = ControlPolicy(model, obs_dim=8).to(device)
policy.adapter.load_state_dict(payload["adapter"])
policy.action_head.load_state_dict(payload["action_head"])
policy = policy.eval()
print(f"model OK — {count_params(model):,} params on {device}")

w = torch.zeros(1, policy.window, 8)
q = policy.action_head(policy.q_features(w))
print("pad-state Q-values:", [f"{v:+.2f}" for v in q[0].tolist()])
```

## Cell 4 — baseline eval of v6 (three-mode matrix, 25 raw episodes)

> **Why three modes (discovered 2026-09-29):** D33's “median −98.3” was produced by
> `episode_breakdown.py`, which never calls `.eval()` — train mode, stochastic Gumbel
> routing. The *deployed* deterministic policy (eval mode + argmax) measures
> **median ≈ −200**. Both numbers are real; they answer different questions. The matrix
> keeps v7 comparable to D33 (mode C) while telling the truth about deployment (mode A).

```python
import sys, statistics
sys.path.insert(0, ".")
import gymnasium as gym, torch
from training.control.obs_adapter import ObsWindow

@torch.no_grad()
def eval_matrix(policy, episodes=25, seed_base=5000):
    """Evaluate A: eval+greedy (deployed), B: eval+sampled, C: train+sampled (D33 mode)."""
    results = {}
    env = gym.make("LunarLander-v3")
    env.action_space.seed(seed_base)
    for name, mode, greedy in (
        ("A eval+greedy   (deployed truth)", "eval", True),
        ("B eval+sampled  (evaluate.py)",   "eval", False),
        ("C train+sampled (D33 comparable)", "train", False),
    ):
        policy.train() if mode == "train" else policy.eval()
        rets = []
        for i in range(episodes):
            obs, _ = env.reset(seed=seed_base + i)
            window = ObsWindow(policy.window, policy.adapter.obs_dim)
            total, done = 0.0, False
            while not done:
                window.push(torch.as_tensor(obs, dtype=torch.float32))
                q = policy.action_head(policy.q_features(window.tensor()))
                if greedy:
                    action = int(q.argmax(dim=-1).item())
                else:
                    action = int(torch.distributions.Categorical(logits=q).sample().item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            rets.append(total)
        results[name] = rets
        print(f"{name}: mean {statistics.mean(rets):+7.1f} "
              f"| median {statistics.median(rets):+7.1f} "
              f"| best {max(rets):+7.1f} | worst {min(rets):+7.1f} "
              f"| > -100: {sum(1 for r in rets if r > -100)}/{len(rets)}", flush=True)
    policy.eval()
    env.close()
    return results

base_matrix = eval_matrix(policy)
# v6 reference (local RTX 3060): A median -200.7 | B mean -224.0 | C median -95.8
# If your Kaggle numbers land near these, the whole chain is verified.
```

## Cell 5 — the consolidation config (the one change that matters)

```python
import sys; sys.path.insert(0, ".")
from training.control.train_dqn import DQNConfig

EPISODES = 1500
EPS_PINNED = 0.02
SEED = 7

config = DQNConfig(
    episodes=EPISODES,
    action_repeat=3,
    eps_start=EPS_PINNED,   # <- flat eps: no 15K-decision re-exploration phase
    eps_end=EPS_PINNED,
    warmup_decisions=0,     # resuming a trained Q — no random warm-up
    # D32 defaults kept: tau=0.005, per_alpha=0.6, beta 0.4 -> 1.0
)
print(config)
assert config.eps_start == config.eps_end == 0.02
assert config.warmup_decisions == 0 and config.tau > 0 and config.per_alpha > 0
```

## Cell 6 — the overnight run (autosaves to /kaggle/working every 30 min)

```python
import sys, time, glob, os
sys.path.insert(0, ".")
import gymnasium as gym, torch

from training.control.shaped_env import ShapedLanderEnv
from training.control.train_dqn import DQNTrainer
from frostbite.utils import seed_everything

WORK = "/kaggle/working"
CKPT_OUT = os.path.join(WORK, "control_dqn_v7.pt")
BUFFER_OUT = os.path.join(WORK, "v7_buffer.pt")
AUTOSAVE_SEC = 30 * 60

seed_everything(SEED)
env = ShapedLanderEnv(gym.make("LunarLander-v3"), gamma=config.gamma)
env.action_space.seed(SEED)
trainer = DQNTrainer(policy, env, config, seed=SEED)
print(f"frame-skip {config.action_repeat} | buffer {config.buffer_size:,} "
      f"| lr {config.lr} | eps pinned {config.eps_end} | tau {config.tau} "
      f"| PER a={config.per_alpha} | shaping ON (train), raw eval")

# -- recovery: if a previous session's output is attached as Input, continue it --
returns_so_far = []
prev_ckpt = sorted(glob.glob("/kaggle/input/**/control_dqn_v7.pt", recursive=True))
prev_buf = sorted(glob.glob("/kaggle/input/**/v7_buffer.pt", recursive=True))
if prev_ckpt:
    saved = torch.load(prev_ckpt[0], map_location=device, weights_only=True)
    for net, key in ((trainer.policy.model, "model"),
                     (trainer.policy.adapter, "adapter"),
                     (trainer.policy.action_head, "action_head")):
        net.load_state_dict(saved[key])
    for net, key in ((trainer.target.model, "model"),
                     (trainer.target.adapter, "adapter"),
                     (trainer.target.action_head, "action_head")):
        net.load_state_dict(saved[key])
    if "optimizer" in saved:
        trainer.optimizer.load_state_dict(saved["optimizer"])
    trainer.decision_step = int(saved.get("decision_step", 0))
    trainer.gradient_steps = int(saved.get("gradient_steps", 0))
    if prev_buf:
        buf = torch.load(prev_buf[0], map_location="cpu", weights_only=True)
        trainer.buffer.buffer.extend(buf["transitions"])
        trainer.buffer.priorities.extend(buf["priorities"])
        print(f"restored buffer: {len(trainer.buffer):,} transitions")
    returns_so_far = list(saved.get("returns", []))
    print(f"RESUMED from previous autosave — {len(returns_so_far)} episodes banked")
else:
    print("no previous autosave attached — starting fresh from v6")

# keep the TOTAL budget at EPISODES across resumed sessions
config.episodes = max(1, config.episodes - len(returns_so_far))
print(f"episode budget this session: {config.episodes}")

# -- autosave between episodes (main thread — no locking worries) --------------
def save_state() -> None:
    """Snapshot weights + optimizer + counters + buffer to /kaggle/working."""
    torch.save(
        {
            "model": trainer.policy.model.state_dict(),
            "adapter": trainer.policy.adapter.state_dict(),
            "action_head": trainer.policy.action_head.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "decision_step": trainer.decision_step,
            "gradient_steps": trainer.gradient_steps,
            "returns": list(returns_so_far),
            "solved": False,
            "protocol": "D34 consolidation (autosave)",
        },
        CKPT_OUT,
    )
    torch.save(
        {
            "transitions": list(trainer.buffer.buffer),
            "priorities": list(trainer.buffer.priorities),
            "alpha": trainer.buffer.alpha,
        },
        BUFFER_OUT,
    )

_last_save = time.time()
_run_episode_orig = trainer._run_episode

def _run_episode_logged(first: bool):
    """Wrap one episode: bank the return, autosave when 30 min have passed."""
    global _last_save
    ret, decisions, loss = _run_episode_orig(first)
    returns_so_far.append(ret)
    if time.time() - _last_save >= AUTOSAVE_SEC:
        save_state()
        _last_save = time.time()
        print(f"[autosave] episode {len(returns_so_far)} | "
              f"{len(trainer.buffer):,} transitions -> {WORK}", flush=True)
    return ret, decisions, loss

trainer._run_episode = _run_episode_logged
print("autosave armed (30 min interval, episode boundary)")

t0 = time.time()
history, solved = trainer.train()  # rich history = this session only
save_state()                        # belt & braces: never end a session unsaved
dt = time.time() - t0
print(f"\ndone in {dt/3600:.2f} h ({dt/60/max(1,len(history)):.1f} min/ep) "
      f"| episodes this session {len(history)} | solved {solved} | "
      f"total curve {len(returns_so_far)} episodes")
```

## Cell 7 — raw eval matrix + curves

```python
import sys, statistics
sys.path.insert(0, ".")
import matplotlib.pyplot as plt

final_matrix = eval_matrix(trainer.policy)
print("\n--- v6 -> v7 ---")
for name in base_matrix:
    b, f = base_matrix[name], final_matrix[name]
    print(f"{name}: median {statistics.median(b):+7.1f} -> {statistics.median(f):+7.1f} "
          f"| mean {statistics.mean(b):+7.1f} -> {statistics.mean(f):+7.1f}")
landings = sum(1 for r in returns_so_far if r >= 0)
print(f"train landings (return >= 0): {landings}/{len(returns_so_far)}")

rets = returns_so_far
k = 20
trailing = [sum(rets[max(0, i-k):i+1]) / len(rets[max(0, i-k):i+1]) for i in range(len(rets))]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(rets, alpha=0.3, lw=0.8, label="episode return")
ax1.plot(trailing, color="tab:red", lw=2, label="trailing-20")
ax1.axhline(0, color="k", lw=0.5); ax1.axhline(200, color="g", ls="--", lw=1, label="SOLVED +200")
ax1.set_xlabel("episode"); ax1.set_ylabel("raw return"); ax1.legend(); ax1.set_title("v7 returns")
ax2.plot([h["loss"] for h in history], color="tab:purple", lw=0.8)
ax2.set_xlabel("episode (this session)"); ax2.set_ylabel("smooth-L1 loss"); ax2.set_title("Q loss")
plt.tight_layout(); plt.show()
```

## Cell 8 — final artifacts to /kaggle/working (becomes the notebook Output on commit)

```python
import os, statistics

def _matrix_summary(matrix):
    """Summarize an eval matrix {label: returns} for the checkpoint payload."""
    return {name: {"mean": statistics.mean(r), "median": statistics.median(r),
                   "best": max(r), "worst": min(r),
                   "gt_-100": sum(1 for x in r if x > -100), "episodes": len(r)}
            for name, r in matrix.items()}

ckpt = {
    "model": trainer.policy.model.state_dict(),
    "adapter": trainer.policy.adapter.state_dict(),
    "action_head": trainer.policy.action_head.state_dict(),
    "optimizer": trainer.optimizer.state_dict(),
    # "returns" spans autosave sessions; "history" (run_dqn.py schema) is this
    # session's rich records — kept separate to avoid duplicated episodes.
    "returns": list(returns_so_far),
    "history": history,
    "solved": solved,
    "config": {"episodes": EPISODES, "eps_start": config.eps_start,
               "eps_end": config.eps_end, "warmup_decisions": config.warmup_decisions,
               "resumed_from": "control_dqn_v6.pt", "protocol": "D34 consolidation"},
    # v6 same-seed/same-mode reference, measured locally on the RTX 3060
    # (path B reproduced on Kaggle to the decimal). Baseline matrix is empty
    # if cell 4 was skipped in this session — the reference above covers it.
    "v6_reference_eval": {"A eval+greedy": {"mean": -198.9, "median": -200.7},
                          "B eval+sampled": {"mean": -224.0, "median": -212.9},
                          "C train+sampled": {"mean": -97.1, "median": -95.8}},
    "baseline_eval": _matrix_summary(globals().get("base_matrix", {})),
    "final_eval": _matrix_summary(globals().get("final_matrix", {})),
}
torch.save(ckpt, os.path.join(WORK, "control_dqn_v7.pt"))  # supersedes the autosave
torch.save({"transitions": list(trainer.buffer.buffer),
            "priorities": list(trainer.buffer.priorities),
            "alpha": trainer.buffer.alpha},
           os.path.join(WORK, "v7_buffer.pt"))
fig.savefig(os.path.join(WORK, "v7_curves.png"), dpi=150)

for f in sorted(os.listdir(WORK)):
    p = os.path.join(WORK, f)
    if os.path.isfile(p):
        print(f"{f:<24} {os.path.getsize(p)/1e6:8.1f} MB")
```

## Cell 9 — optional: push checkpoints to a private Kaggle dataset mid-run

Only if you want persistence beyond `/kaggle/working` (e.g. paranoid about interactive-session
death). Add-ons → Secrets → add `KAGGLE_USERNAME` / `KAGGLE_KEY` (kaggle.com → Account →
Create New API Token) first. Change `YOUR_USERNAME` to match.

```python
# run interactively whenever you want an off-machine copy (also safe inside cell 6's autosave)
import os, shutil, tempfile
os.environ["KAGGLE_USERNAME"] = UserSecretsClient().get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = UserSecretsClient().get_secret("KAGGLE_KEY")

import kagglehub
tmp = tempfile.mkdtemp()
shutil.copy2(os.path.join(WORK, "control_dqn_v7.pt"), tmp)
shutil.copy2(os.path.join(WORK, "v7_buffer.pt"), tmp)
kagglehub.dataset_upload("YOUR_USERNAME/frostbite-v7", tmp)
print("pushed to Kaggle Datasets")
```

---

## If a session dies mid-run

1. Open the notebook again. If you ran via **Save & Run All**, the partial autosave is in the
   failed/completed version's Output — download it, or attach that version's output as Input.
2. *Add Input* → *Your Work* → this notebook's previous version **Output** (contains
   `control_dqn_v7.pt` + `v7_buffer.pt`). Keep the `frostbite-v6` dataset attached too.
3. *Run all*. Cell 6 finds the autosave, restores **weights + Adam state + PER buffer +
   the return curve + decision/gradient counters**, and shrinks the episode budget to the
   remaining total. With ε pinned there's no exploration-schedule offset to correct —
   a resumed run is a clean continuation, not a re-baseline.

## Honest expectations (recalibrated 2026-09-29 after the eval-mode discovery)

D33's −98.3 was a train-mode number. The deployed greedy policy of v6 is
**median ≈ −200**; training acts in exactly that regime (the DQN trainer runs the
policy in eval mode with ε=0.02), so consolidation polishs the policy you deploy:

| Outcome | Likelihood |
| --- | --- |
| Deployed (A) median −200 → −120…−80 | plausible — trailing-20 slope supports consolidation gains |
| A-mode landings appear (v6 greedy: 0 clean) | plausible |
| C-mode (D33-comparable) −98 → −60…−40 | good — the regime D32's fixes were built for |
| `SOLVED` (+200) from length alone | unlikely — needs a qualitative jump, not more epochs |
| Nothing improves / slight decay | possible — greedy imperfection may cap gains |

Also watch for: does the A↔C gap close? If consolidation tightens Q, argmax and
> sampled routing should converge — that would be its own finding.

## Timing notes

- On the RTX 3060 this budget was ≈4–5 h; Kaggle's T4/P100 should be similar — the CfC
  python time-loop is latency-bound, not compute-bound.
- GPU sessions cap at 12 h (quota ~30 h/week): one 1500-episode run fits with room to spare.
- Bring the artifacts home: notebook version → *Output* → download, or push via cell 9.
