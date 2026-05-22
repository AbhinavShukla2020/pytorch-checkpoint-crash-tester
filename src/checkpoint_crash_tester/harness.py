from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from checkpoint_crash_tester.failure import PHASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline and injected DDP checkpoint trials")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fail-rank", type=int, default=1)
    parser.add_argument("--fail-step", type=int, default=10)
    parser.add_argument("--fail-phase", choices=sorted(PHASES), default="after-rank-write")
    return parser.parse_args()


def command(args: argparse.Namespace, run_dir: Path, attempt: str, inject: bool) -> list[str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    result = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node-rank=0",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        f"--nproc-per-node={args.nproc_per_node}",
        "-m",
        "checkpoint_crash_tester.train",
        "--run-dir",
        str(run_dir),
        "--steps",
        str(args.steps),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--batch-size",
        str(args.batch_size),
        "--dataset-size",
        str(args.dataset_size),
        "--seed",
        str(args.seed),
        "--attempt",
        attempt,
    ]
    if inject:
        result.extend(
            [
                "--fail-rank",
                str(args.fail_rank),
                "--fail-step",
                str(args.fail_step),
                "--fail-phase",
                args.fail_phase,
            ]
        )
    return result


def run_process(arguments: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    if sys.platform == "darwin":
        environment.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            arguments,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return_code = 124
    return return_code, time.perf_counter() - started


def read_results(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((run_dir / "results").glob("rank_*.json"))
    return [json.loads(path.read_text()) for path in paths]


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted((run_dir / "events").glob("rank_*.jsonl")):
        for line in path.read_text().splitlines():
            events.append(json.loads(line))
    return events


def compare_results(
    baseline: list[dict[str, Any]], recovered: list[dict[str, Any]]
) -> list[str]:
    differences: list[str] = []
    if len(baseline) != len(recovered):
        return [f"result rank count differs: {len(baseline)} != {len(recovered)}"]
    for expected, actual in zip(baseline, recovered, strict=True):
        rank = expected["rank"]
        for key in ("rank", "world_size", "global_step", "optimizer_step", "model_digest"):
            if expected[key] != actual[key]:
                differences.append(f"rank {rank} {key}: {expected[key]} != {actual[key]}")
        if expected["logical_sample_ids"] != actual["logical_sample_ids"]:
            differences.append(f"rank {rank} logical sample sequence differs")
    return differences


def validate(args: argparse.Namespace) -> None:
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise SystemExit(f"work directory must be absent or empty: {args.work_dir}")
    if args.nproc_per_node < 2:
        raise SystemExit("nproc-per-node must be at least two")
    if not 0 <= args.fail_rank < args.nproc_per_node:
        raise SystemExit("fail-rank must be inside the process group")
    if args.fail_step <= 0 or args.fail_step > args.steps:
        raise SystemExit("fail-step must be inside the training run")
    if args.fail_step % args.checkpoint_every != 0:
        raise SystemExit("fail-step must be a checkpoint step")


def main() -> None:
    args = parse_args()
    validate(args)
    baseline_dir = args.work_dir / "baseline"
    crash_dir = args.work_dir / "crash"

    baseline_code, baseline_seconds = run_process(
        command(args, baseline_dir, "baseline", inject=False),
        args.work_dir / "logs" / "baseline.log",
    )
    if baseline_code != 0:
        raise SystemExit(f"baseline failed; inspect {args.work_dir / 'logs' / 'baseline.log'}")

    crash_code, crash_seconds = run_process(
        command(args, crash_dir, "injected", inject=True),
        args.work_dir / "logs" / "injected.log",
    )
    if crash_code == 0:
        raise SystemExit("injected run unexpectedly completed without a worker failure")
    if not (crash_dir / "injection_marker.json").is_file():
        raise SystemExit("injected run failed without reaching the configured failpoint")

    recovery_code, recovery_seconds = run_process(
        command(args, crash_dir, "recovery", inject=False),
        args.work_dir / "logs" / "recovery.log",
    )
    if recovery_code != 0:
        raise SystemExit(f"recovery failed; inspect {args.work_dir / 'logs' / 'recovery.log'}")

    baseline = read_results(baseline_dir)
    recovered = read_results(crash_dir)
    differences = compare_results(baseline, recovered)
    events = read_events(crash_dir)
    failed_steps = [
        int(event["step"])
        for event in events
        if event.get("attempt") == "injected" and event.get("event") == "step"
    ]
    recovery_starts = [
        int(event["resumed_from_step"])
        for event in events
        if event.get("attempt") == "recovery" and event.get("event") == "start"
    ]
    furthest_failed_step = max(failed_steps, default=0)
    resumed_from_step = min(recovery_starts) if recovery_starts else 0
    report = {
        "equivalent": not differences,
        "differences": differences,
        "configuration": {
            "world_size": args.nproc_per_node,
            "steps": args.steps,
            "checkpoint_every": args.checkpoint_every,
            "batch_size_per_rank": args.batch_size,
            "dataset_size": args.dataset_size,
            "seed": args.seed,
            "failure": {
                "rank": args.fail_rank,
                "step": args.fail_step,
                "phase": args.fail_phase,
            },
        },
        "timing_seconds": {
            "baseline": baseline_seconds,
            "failed_attempt": crash_seconds,
            "restart_to_completion": recovery_seconds,
        },
        "recovery": {
            "furthest_failed_step": furthest_failed_step,
            "resumed_from_step": resumed_from_step,
            "replayed_optimizer_steps": max(0, furthest_failed_step - resumed_from_step),
            "failed_process_returncode": crash_code,
        },
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
