from __future__ import annotations

import math
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class StatefulDistributedSampler(Sampler[int]):
    """Distributed sampler whose epoch-local cursor can be checkpointed."""

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("rank must identify one of num_replicas workers")
        self.dataset_size = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        if drop_last:
            self.num_samples = self.dataset_size // num_replicas
        else:
            self.num_samples = math.ceil(self.dataset_size / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.epoch = 0
        self.offset = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        if self.drop_last:
            indices = indices[: self.total_size]
        elif len(indices) < self.total_size:
            indices += indices[: self.total_size - len(indices)]
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices[self.offset :])

    def __len__(self) -> int:
        return self.num_samples - self.offset

    def advance(self, count: int) -> None:
        if count < 0 or self.offset + count > self.num_samples:
            raise ValueError("sampler advance exceeds the current epoch")
        self.offset += count

    def next_epoch(self) -> None:
        if self.offset != self.num_samples:
            raise ValueError("cannot advance epoch before consuming every rank-local sample")
        self.epoch += 1
        self.offset = 0

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "epoch": self.epoch,
            "offset": self.offset,
            "seed": self.seed,
            "dataset_size": self.dataset_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "drop_last": self.drop_last,
        }

    def load_state_dict(self, state: dict[str, int | bool]) -> None:
        expected = {
            "seed": self.seed,
            "dataset_size": self.dataset_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "drop_last": self.drop_last,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"sampler state mismatch for {key}")
        epoch, offset = int(state["epoch"]), int(state["offset"])
        if epoch < 0 or not 0 <= offset <= self.num_samples:
            raise ValueError("sampler cursor is invalid")
        self.epoch = epoch
        self.offset = offset
