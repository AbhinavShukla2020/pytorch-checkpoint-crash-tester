import json
import random
from pathlib import Path

import numpy as np
import torch

from checkpoint_crash_tester.checkpoint import (
    DistributedCheckpointer,
    capture_rng_state,
    restore_rng_state,
)


def make_model_and_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    return model, optimizer


def test_incomplete_newer_directory_is_ignored(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    checkpointer = DistributedCheckpointer(tmp_path, rank=0, world_size=1, barrier=lambda: None)
    saved = checkpointer.save(
        step=4,
        model=model,
        optimizer=optimizer,
        sampler_state={"epoch": 0, "offset": 8},
        logical_sample_ids=[1, 2, 3],
    )
    partial = tmp_path / "step_00000008"
    partial.mkdir()
    (partial / "rank_00000.pt").write_bytes(b"partial")

    latest = checkpointer.latest_complete()
    assert latest is not None
    assert latest.step == 4
    assert latest.directory == saved.directory


def test_corrupt_rank_file_invalidates_manifest(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    checkpointer = DistributedCheckpointer(tmp_path, rank=0, world_size=1, barrier=lambda: None)
    saved = checkpointer.save(
        step=2,
        model=model,
        optimizer=optimizer,
        sampler_state={"epoch": 0, "offset": 4},
        logical_sample_ids=[4, 5],
    )
    with (saved.directory / "rank_00000.pt").open("ab") as handle:
        handle.write(b"corruption")
    assert checkpointer.latest_complete() is None


def test_manifest_describes_stable_rank_file(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    checkpointer = DistributedCheckpointer(tmp_path, rank=0, world_size=1, barrier=lambda: None)
    saved = checkpointer.save(
        step=3,
        model=model,
        optimizer=optimizer,
        sampler_state={"epoch": 0, "offset": 6},
        logical_sample_ids=[1],
    )
    manifest = json.loads((saved.directory / "manifest.json").read_text())
    assert manifest["step"] == 3
    assert manifest["files"][0]["name"] == "rank_00000.pt"
    assert not list(saved.directory.glob("*.tmp"))


def test_rng_state_restores_all_cpu_streams() -> None:
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.random()
    np.random.random()
    torch.rand(())
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected
