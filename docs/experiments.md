# Experiment design

The harness runs a baseline and an injected-failure trial from identical seeds.
The final comparison includes:

- SHA-256 digest of sorted model tensors.
- Optimizer step for every parameter carrying Adam state.
- Per-rank sequence of logical sample IDs.
- Final global optimizer-step count.

The failed attempt and recovery attempt append structured events to separate
JSONL files. The report derives replayed work from the largest step observed
before failure minus the selected recovery step. Restart latency is measured
from launching the recovery `torchrun` process until it reports completion.

Record the following next to any published result:

- Git commit and clean/dirty working-tree state.
- PyTorch, Python, CUDA, NCCL, and operating-system versions.
- Backend, world size, model dimensions, batch size, and checkpoint interval.
- Storage medium and whether the filesystem is local or networked.
- Injected rank, step, phase, and process exit status.

Avoid comparing restart times across machines without normalizing checkpoint
size and storage performance.
