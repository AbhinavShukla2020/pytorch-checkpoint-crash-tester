from checkpoint_crash_tester.sampler import StatefulDistributedSampler


class Dataset:
    def __len__(self) -> int:
        return 23


def test_resumed_sampler_matches_uninterrupted_suffix() -> None:
    original = StatefulDistributedSampler(Dataset(), num_replicas=2, rank=1, seed=19)
    full_sequence = list(original)

    interrupted = StatefulDistributedSampler(Dataset(), num_replicas=2, rank=1, seed=19)
    first = list(interrupted)[:5]
    interrupted.advance(len(first))
    state = interrupted.state_dict()

    resumed = StatefulDistributedSampler(Dataset(), num_replicas=2, rank=1, seed=19)
    resumed.load_state_dict(state)
    assert first + list(resumed) == full_sequence


def test_ranks_receive_different_positions() -> None:
    rank_zero = StatefulDistributedSampler(Dataset(), num_replicas=2, rank=0, seed=3)
    rank_one = StatefulDistributedSampler(Dataset(), num_replicas=2, rank=1, seed=3)
    assert list(rank_zero) != list(rank_one)
    assert len(rank_zero) == len(rank_one)
