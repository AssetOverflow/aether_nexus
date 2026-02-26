---
description: Summarize Distiller REM cycle config and hot/cold pool split from the codebase
allowed-tools: Read, Grep
---

Read the following files and produce a Distiller status report:

- `nexus-core/src/distiller.rs` — REM cycle interval, entropy threshold, distillation logic
- `nexus-core/src/types.rs` — `SparseCode` definition, `FabricLayout` pool regions
- `nexus-core/src/fabric.rs` — hot/cold pool sizes and offsets

Report the following:

**REM Cycle Config**
- Wake interval (seconds)
- Entropy threshold for hot → cold migration
- Max blocks distilled per cycle

**Pool Layout**
- Hot KV pool: offset range + size in GB
- Cold KV pool: offset range + size in GB
- Learned dictionary: offset range + size in MB
- Expected compression ratio (hot f16 size → cold SparseCode size)

**SparseCode Format**
- Struct size (should be exactly 16 bytes)
- Dictionary size per KV head (should be 512 vectors)
- Reconstruction accuracy: O(4) sparse dot product

**Health Checks**
- Confirm `SparseCode` is 16 bytes (`size_of` reasoning)
- Confirm WAL flush interval matches the 300ms target
- Flag any TODO/FIXME comments in distiller.rs that indicate incomplete implementation
- Note any hardcoded magic numbers that should be constants

End with: ✓ Distiller looks healthy / ⚠ Issues found (list them).
