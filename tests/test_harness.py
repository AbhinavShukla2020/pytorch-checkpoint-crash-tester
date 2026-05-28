from checkpoint_crash_tester.harness import compare_results


def result(rank: int, digest: str = "abc") -> dict:
    return {
        "rank": rank,
        "world_size": 2,
        "global_step": 10,
        "optimizer_step": 10,
        "model_digest": digest,
        "logical_sample_ids": [rank, rank + 2],
    }


def test_equal_results_have_no_differences() -> None:
    baseline = [result(0), result(1)]
    assert compare_results(baseline, [result(0), result(1)]) == []


def test_model_difference_is_reported() -> None:
    differences = compare_results([result(0)], [result(0, digest="changed")])
    assert "model_digest" in differences[0]
