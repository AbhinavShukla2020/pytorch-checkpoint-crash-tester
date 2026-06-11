# Checkpoint publication protocol

Each checkpoint is a directory named for its completed optimizer step. Saving is
a small distributed commit protocol:

1. Every rank serializes model, optimizer, RNG, sampler, and audit state to a
   unique temporary file in the target directory.
2. The process flushes and `fsync`s the file, then atomically renames it to the
   stable rank filename.
3. All ranks enter a process-group barrier.
4. Rank zero hashes every rank file, writes a temporary manifest, flushes it,
   and atomically renames it to `manifest.json`.
5. A final barrier prevents a worker from racing ahead while publication is
   still completing.

Discovery requires the manifest, expected world size, every listed file, and a
matching SHA-256 digest. A directory that fails any check is incomplete and is
not a recovery candidate.

## Failure phases

The harness can kill one rank at four points:

- `before-rank-write`: before any bytes for that rank are serialized.
- `after-rank-write`: after the stable rank file exists but before the barrier.
- `before-manifest`: rank zero has observed every rank file but has not published.
- `after-manifest`: the checkpoint is committed, but workers have not left the
  final barrier.

A persistent marker makes injection one-shot. Restarting with the same work
directory cannot repeatedly kill the worker at the same failpoint.

## Assumptions

Atomic rename and file durability are scoped to one filesystem. The protocol
does not claim transactional behavior across object stores or network mounts
with weaker consistency. Production deployments should use the storage-specific
commit primitives provided by their checkpointing system.
