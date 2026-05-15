from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from checkpoint_crash_tester.checkpoint import (
    DistributedCheckpointer,
    model_digest,
    optimizer_step,
)
from checkpoint_crash_tester.failure import PHASES, FailurePlan
from checkpoint_crash_tester.sampler import StatefulDistributedSampler
from checkpoint_crash_tester.workload import DeterministicRegressionDataset, TinyRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic DDP checkpoint workload")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument("--attempt", default="run")
    parser.add_argument("--fail-rank", type=int)
    parser.add_argument("--fail-step", type=int)
    parser.add_argument("--fail-phase", choices=sorted(PHASES))
    return parser.parse_args()


def append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def setup_process_group(backend: str) -> tuple[int, int, torch.device]:
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)
    return rank, world_size, device


def failure_plan(args: argparse.Namespace) -> FailurePlan | None:
    supplied = (args.fail_rank, args.fail_step, args.fail_phase)
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise ValueError("fail-rank, fail-step, and fail-phase must be provided together")
    return FailurePlan(
        rank=args.fail_rank,
        step=args.fail_step,
        phase=args.fail_phase,
        marker=args.run_dir / "injection_marker.json",
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.checkpoint_every <= 0 or args.batch_size <= 0:
        raise ValueError("steps, checkpoint interval, and batch size must be positive")
    rank, world_size, device = setup_process_group(args.backend)
    event_path = args.run_dir / "events" / f"rank_{rank:05d}.jsonl"

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    dataset = DeterministicRegressionDataset(args.dataset_size, args.dimensions, seed=args.seed)
    sampler = StatefulDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=args.seed + 10_000,
    )
    module = TinyRegressor(args.dimensions).to(device)
    distributed_model = DistributedDataParallel(module)
    optimizer = torch.optim.AdamW(distributed_model.parameters(), lr=1e-3)
    checkpointer = DistributedCheckpointer(
        args.run_dir / "checkpoints", rank, world_size, dist.barrier
    )

    global_step = 0
    logical_sample_ids: list[int] = []
    restored = checkpointer.load_latest(module, optimizer)
    if restored is not None:
        global_step, sampler_state, logical_sample_ids = restored
        sampler.load_state_dict(sampler_state)

    append_event(
        event_path,
        {
            "event": "start",
            "attempt": args.attempt,
            "rank": rank,
            "resumed_from_step": global_step,
            "time_ns": time.time_ns(),
        },
    )
    plan = failure_plan(args)

    while global_step < args.steps:
        iterator = iter(sampler)
        while global_step < args.steps:
            indices = list(islice(iterator, args.batch_size))
            if not indices:
                sampler.next_epoch()
                break

            ids = torch.tensor(indices, dtype=torch.int64)
            features = dataset.features[ids].to(device)
            targets = dataset.targets[ids].to(device)
            predictions = distributed_model(features)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sampler.advance(len(indices))
            logical_sample_ids.extend(indices)
            global_step += 1
            append_event(
                event_path,
                {
                    "event": "step",
                    "attempt": args.attempt,
                    "rank": rank,
                    "step": global_step,
                    "sample_ids": indices,
                    "loss": float(loss.detach().cpu()),
                    "time_ns": time.time_ns(),
                },
            )

            if global_step % args.checkpoint_every == 0 or global_step == args.steps:
                checkpointer.save(
                    step=global_step,
                    model=module,
                    optimizer=optimizer,
                    sampler_state=sampler.state_dict(),
                    logical_sample_ids=logical_sample_ids,
                    fault=(
                        (lambda step, phase: plan.maybe_kill(rank, step, phase))
                        if plan is not None
                        else None
                    ),
                )

    dist.barrier()
    write_json_atomic(
        args.run_dir / "results" / f"rank_{rank:05d}.json",
        {
            "rank": rank,
            "world_size": world_size,
            "global_step": global_step,
            "optimizer_step": optimizer_step(optimizer),
            "model_digest": model_digest(module),
            "logical_sample_ids": logical_sample_ids,
        },
    )
    append_event(
        event_path,
        {
            "event": "complete",
            "attempt": args.attempt,
            "rank": rank,
            "step": global_step,
            "time_ns": time.time_ns(),
        },
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
