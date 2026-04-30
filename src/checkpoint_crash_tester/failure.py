from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path

PHASES = {"before-rank-write", "after-rank-write", "before-manifest", "after-manifest"}


@dataclass(frozen=True)
class FailurePlan:
    rank: int
    step: int
    phase: str
    marker: Path

    def __post_init__(self) -> None:
        if self.rank < 0 or self.step <= 0 or self.phase not in PHASES:
            raise ValueError("invalid failure plan")

    def maybe_kill(self, rank: int, step: int, phase: str) -> None:
        if (rank, step, phase) != (self.rank, self.step, self.phase):
            return
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "rank": rank, "step": step, "phase": phase}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.kill(os.getpid(), signal.SIGKILL)
