import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_two_rank_crash_and_resume_matches_baseline(tmp_path: Path) -> None:
    work_dir = tmp_path / "experiment"
    command = [
        sys.executable,
        "-m",
        "checkpoint_crash_tester.harness",
        "--work-dir",
        str(work_dir),
        "--nproc-per-node",
        "2",
        "--steps",
        "6",
        "--checkpoint-every",
        "2",
        "--batch-size",
        "4",
        "--dataset-size",
        "64",
        "--fail-rank",
        "1",
        "--fail-step",
        "4",
        "--fail-phase",
        "after-rank-write",
    ]
    environment = os.environ.copy()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"harness failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    report = json.loads((work_dir / "report.json").read_text())
    assert report["equivalent"] is True
    assert report["recovery"]["resumed_from_step"] == 2
    assert report["recovery"]["replayed_optimizer_steps"] == 2
