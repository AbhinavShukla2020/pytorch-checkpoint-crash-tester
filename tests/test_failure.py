import os
from pathlib import Path

from checkpoint_crash_tester.failure import FailurePlan


def test_failure_marker_makes_injection_one_shot(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, signal: calls.append((pid, signal)))
    plan = FailurePlan(rank=1, step=5, phase="after-rank-write", marker=tmp_path / "marker")
    plan.maybe_kill(1, 5, "after-rank-write")
    plan.maybe_kill(1, 5, "after-rank-write")
    assert len(calls) == 1
    assert plan.marker.is_file()
