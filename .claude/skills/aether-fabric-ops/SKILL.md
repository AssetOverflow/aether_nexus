---
name: Aether Fabric Operations
description: Activate this skill when discussing or modifying the Unified Fabric (.aether file), FabricLayout, WAL flush behavior, Ed25519 signature verification, mmap semantics, the Distiller REM cycle, or KV pool management (hot/cold). Also activate when the topic involves the .aether file format, pool eviction policy, or Fabric recovery from WAL.
version: 1.0.0
---

# Aether Fabric Operations — AetherNexus

## What is the Fabric?

The Fabric is a single memory-mapped file (`brain.aether`) that contains the complete state of the inference engine: model weights, KV caches, persona looms, learned dictionary, and the holographic trace. All components share this one mmap region — no IPC, no serialization.

**File:** `nexus-core/src/fabric.rs`

## FabricLayout Memory Map

```
┌─────────────────────────────────────────┐ offset 0
│  Header (magic: u64, version: u32)      │ 16 bytes
├─────────────────────────────────────────┤
│  Weights Region (f16/Q4)                │ model-dependent
├─────────────────────────────────────────┤
│  Hot KV Pool (exact f16)                │ ~6-8 GB, page-aligned
├─────────────────────────────────────────┤
│  Cold KV Pool (SparseCode)              │ ~0.5-1 GB, page-aligned
├─────────────────────────────────────────┤
│  Learned Dictionary (512 vec/KV head)   │ <100 MB
├─────────────────────────────────────────┤
│  Loom Descriptors (4 personas)          │ fixed
├─────────────────────────────────────────┤
│  Radix Metadata (prefix dedup)          │ fixed
├─────────────────────────────────────────┤
│  Observation Buffers (Cortex writes)    │ per-capability, fixed
├─────────────────────────────────────────┤
│  Holographic Trace (replay log)         │ circular buffer
├─────────────────────────────────────────┤
│  Ed25519 Signature (64 bytes)           │ LAST — covers all above
└─────────────────────────────────────────┘
```

**Critical invariants:**
- `FabricLayout` is `#[repr(C, align(16384))]` — 16 KB page alignment.
- The signature covers every byte from offset 0 through the end of the holographic trace.
- Adding fields to `FabricLayout` changes offsets and **invalidates all existing .aether files**.

## WAL (Write-Ahead Log)

- Flush interval: 300 ms (configurable).
- WAL records mutations before applying them to the mmap region.
- On crash recovery: replay WAL forward, then verify Ed25519 signature.
- **Maximum data loss**: 300 ms worth of KV updates.

```rust
// fabric.rs: correct flush order
fabric.wal_flush()?;           // 1. flush WAL to disk
fabric.sign_and_persist()?;    // 2. recompute + write Ed25519 signature
```

Never reverse this order — a crashed signature write leaves the file unverified but the WAL allows replay.

## Ed25519 Signature

- Algorithm: Ed25519 via `ring` crate.
- The keypair is generated at genesis and stored externally (never in the .aether file).
- On boot: `fabric.verify_signature(&public_key)?` — fails fast if tampered.
- The signature is the **last 64 bytes** of the file.

## Hot Pool ↔ Cold Pool (Distiller)

| Pool | Format | Size | Accuracy |
|------|--------|------|----------|
| Hot KV | Exact f16 | ~6-8 GB | Lossless |
| Cold KV | SparseCode (16 bytes) | ~0.5-1 GB | ~O(4) reconstruction |

**Distiller REM cycle** (`distiller.rs`):
1. Background Tokio task wakes every N seconds (default 30s).
2. Evaluates entropy of hot pool blocks (high entropy → frequently changing → keep hot).
3. Low-entropy blocks → pack to `SparseCode` → write to cold pool → free hot slot.
4. Cold-path blocks are reconstructed on-the-fly in `weaver_kernel.metal` without ever being decompressed to f16.

## Radix Tree (Prefix Dedup)

The radix metadata region deduplicates KV prefixes across the 4 persona looms. Shared prefixes are stored once and referenced by pointer offset within the Fabric. When modifying the loom or radix structures, be careful not to invalidate live references.

## Common Pitfalls

| Pitfall | Consequence | Prevention |
|---------|-------------|-----------|
| Adding padding to FabricLayout | Signature mismatch on existing .aether files | Always use `#[repr(C)]` + size assertions |
| Mutating mmap after sign step | Signature invalid until next flush | Never write to mmap outside WAL transaction |
| Sending mmap raw ptr across threads | Data race | Wrap in `Arc<Mutex<>>` or use dedicated write thread |
| Forgetting to flush before sign | Stale WAL entries appear inconsistent | Always: wal_flush → sign_and_persist |
| Changing KV head count in config | Cold pool SparseCode indices invalid | .aether files are model-specific — use genesis for new models |

## Genesis

A new `.aether` file is created by the genesis procedure in `main.rs`:
1. Allocate file to required size.
2. mmap the file.
3. Write header magic + version.
4. Load model weights into weights region.
5. Zero-initialize all pools and buffers.
6. Generate Ed25519 keypair.
7. Sign the initial state.

The resulting file is self-contained and verifiable without any external state.
