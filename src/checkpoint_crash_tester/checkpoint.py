from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompleteCheckpoint:
    step: int
    directory: Path
    manifest: dict[str, Any]


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def optimizer_step(optimizer: Optimizer) -> int:
    steps: list[int] = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        value = state["step"]
        steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
    return max(steps, default=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(value: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


class DistributedCheckpointer:
    def __init__(
        self,
        root: Path,
        rank: int,
        world_size: int,
        barrier: Callable[[], None],
    ) -> None:
        self.root = root
        self.rank = rank
        self.world_size = world_size
        self.barrier = barrier
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        sampler_state: dict[str, int | bool],
        logical_sample_ids: list[int],
        fault: Callable[[int, str], None] | None = None,
    ) -> CompleteCheckpoint:
        checkpoint_dir = self.root / f"step_{step:08d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trigger = fault or (lambda _step, _phase: None)

        trigger(step, "before-rank-write")
        rank_file = checkpoint_dir / f"rank_{self.rank:05d}.pt"
        _atomic_torch_save(
            {
                "format_version": 1,
                "step": step,
                "rank": self.rank,
                "world_size": self.world_size,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng_state(),
                "sampler": sampler_state,
                "logical_sample_ids": logical_sample_ids,
            },
            rank_file,
        )
        trigger(step, "after-rank-write")
        self.barrier()
        trigger(step, "before-manifest")
        self.barrier()

        if self.rank == 0:
            files = []
            for rank in range(self.world_size):
                path = checkpoint_dir / f"rank_{rank:05d}.pt"
                if not path.is_file():
                    raise CheckpointError(f"rank file missing before publication: {path}")
                files.append({"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size})
            _atomic_json(
                {
                    "format_version": 1,
                    "step": step,
                    "world_size": self.world_size,
                    "files": files,
                },
                checkpoint_dir / "manifest.json",
            )

        self.barrier()
        trigger(step, "after-manifest")
        self.barrier()
        checkpoint = self._validate(checkpoint_dir)
        if checkpoint is None:
            raise CheckpointError(f"published checkpoint did not validate: {checkpoint_dir}")
        return checkpoint

    def latest_complete(self) -> CompleteCheckpoint | None:
        candidates = sorted(self.root.glob("step_*"), reverse=True)
        for directory in candidates:
            checkpoint = self._validate(directory)
            if checkpoint is not None:
                return checkpoint
        return None

    def load_latest(
        self, model: nn.Module, optimizer: Optimizer
    ) -> tuple[int, dict[str, int | bool], list[int]] | None:
        checkpoint = self.latest_complete()
        if checkpoint is None:
            return None
        rank_file = checkpoint.directory / f"rank_{self.rank:05d}.pt"
        payload = torch.load(rank_file, map_location="cpu", weights_only=False)
        if payload["rank"] != self.rank or payload["world_size"] != self.world_size:
            raise CheckpointError("rank payload does not match the current process group")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng"])
        return int(payload["step"]), payload["sampler"], list(payload["logical_sample_ids"])

    def _validate(self, directory: Path) -> CompleteCheckpoint | None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
            step = int(manifest["step"])
            if manifest["format_version"] != 1 or int(manifest["world_size"]) != self.world_size:
                return None
            if directory.name != f"step_{step:08d}" or len(manifest["files"]) != self.world_size:
                return None
            for entry in manifest["files"]:
                path = directory / entry["name"]
                if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
                    return None
                if _sha256(path) != entry["sha256"]:
                    return None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return None
        return CompleteCheckpoint(step=step, directory=directory, manifest=manifest)
