.PHONY: test lint experiment

test:
	python3 -m pytest

lint:
	python3 -m ruff check src tests

experiment:
	checkpoint-crash-test --work-dir runs/demo --nproc-per-node 2 --steps 20 --checkpoint-every 5 --fail-rank 1 --fail-step 10 --fail-phase after-rank-write
