"""P6.U2 tests: trajectory collection + warm-start mechanics (tiny scale)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from frostbite.model import FrostbiteModel
from frostbite.utils import count_params, seed_everything
from training.control.collect import TrajectoryDataset, collect_random_trajectories
from training.control.obs_adapter import ObsAdapter
from training.control.warm_start import WarmStartTrainer


def test_collect_shapes_and_determinism():
    a = collect_random_trajectories(n_episodes=2, window=8, seed=5)
    b = collect_random_trajectories(n_episodes=2, window=8, seed=5)
    assert a.windows.shape[1:] == (8, 8)  # (N, T, obs_dim)
    assert a.targets.shape[1:] == (8,)
    assert torch.equal(a.windows, b.windows)  # seeded -> reproducible


def test_dataset_len_getitem():
    ds = TrajectoryDataset(torch.zeros(4, 8, 8), torch.ones(4, 8))
    assert len(ds) == 4
    window, target = ds[0]
    assert window.shape == (8, 8) and target.shape == (8,)


def test_warm_start_loss_decreases(arch_config):
    """The warm-start loop reduces next-obs loss on a tiny config."""
    seed_everything(0)
    dataset = collect_random_trajectories(n_episodes=2, window=8, seed=5)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = FrostbiteModel(arch_config)
    adapter = ObsAdapter(8, arch_config.embedding.sensor_dim, window=8)
    trainer = WarmStartTrainer(model, adapter, lr=3e-3)

    first = trainer.epoch(loader)
    for _ in range(2):
        last = trainer.epoch(loader)
    assert last < first, f"warm-start did not help: {first:.5f} -> {last:.5f}"


def test_control_additions_within_cap(arch_config):
    """Model + adapter + action head stay under the 30M cap."""
    from training.control.policy_wrapper import ControlPolicy

    policy = ControlPolicy(FrostbiteModel(arch_config), obs_dim=8)
    total = (
        count_params(policy.model)
        + count_params(policy.adapter)
        + count_params(policy.action_head)
    )
    assert total <= arch_config.param_cap
