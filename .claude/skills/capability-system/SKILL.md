---
name: Capability System
description: Activate this skill when discussing or modifying the Cortex dispatcher (cortex.rs), the Capability trait (capability.rs), CapabilityId variants (types.rs), or observation buffer slices in FabricLayout. Also activate when reasoning about the action tensor → dispatch → handler → result pipeline, or when reviewing capability args/result type design.
version: 1.0.0
---

# Capability System — AetherNexus

## Overview

The Unified Capability Cortex is the bridge between GPU inference and native Rust actions. When the Weaver decodes an action token, it produces an **action tensor** identifying which capability to invoke and embedding the arguments. The Cortex dispatches zero-copy to the appropriate Rust handler and writes the result into a pre-allocated **observation buffer** slice in the Fabric.

**Files:**
- `nexus-core/src/capability.rs` — `Capability` trait + dispatch macro
- `nexus-core/src/cortex.rs` — dispatch loop + all handler implementations
- `nexus-core/src/types.rs` — `CapabilityId` enum

## The Dispatch Pipeline

```
Weaver GPU output
    └─► action tensor (CapabilityId u32 + args bytes)
            └─► Cortex::dispatch(&self, id: CapabilityId, args: &[u8]) -> &[u8]
                    └─► bytemuck::from_bytes::<Args>(args)
                            └─► handler(args) -> Result
                                    └─► bytemuck::bytes_of(&result)
                                            └─► write to FabricLayout::observation_buffers[capability]
```

Everything is zero-copy: args bytes come directly from the action tensor buffer in the mmap region; results are written directly to the observation buffer slice in the same mmap region.

## Capability Trait

```rust
// capability.rs
pub trait Capability: Pod + Zeroable {
    type Args: Pod + Zeroable;
    type Result: Pod + Zeroable;
    fn invoke(args: &Self::Args) -> Self::Result;
}
```

**`Pod + Zeroable` requirements for Args and Result:**
- `#[repr(C)]` — no compiler-reordered fields
- No padding bytes — use explicit padding fields (`_pad: [u8; N]`) if needed
- No references, pointers, or heap types
- All bit patterns must be valid (no enums with invalid discriminants as fields)
- Fixed, known size at compile time

## CapabilityId Enum

```rust
// types.rs
#[repr(u32)]
pub enum CapabilityId {
    CargoCheck   = 0,
    GitStatus    = 1,
    VectorSearch = 2,
    TensorRegex  = 3,
    SafeEval     = 4,
    // Next: 5, 6, 7 ...
}
```

**Rules:**
- Always `#[repr(u32)]` — must match the action tensor encoding
- Values are **permanent** — never reuse a retired discriminant
- Exhaustive matching in `cortex.rs` — adding a variant requires a new match arm

## Built-in Capabilities

| CapabilityId | Args size | Result size | Description |
|-------------|-----------|-------------|-------------|
| `CargoCheck` | 128 bytes (path) | 512 bytes (exit + stderr) | Runs `cargo check` in a workspace path |
| `GitStatus` | 0 bytes | 48 bytes (branch hash + dirty flag) | Current git HEAD + dirty status |
| `VectorSearch` | 64 bytes (embedding + k) | 256 bytes (top-K indices + scores) | Query Qwen3 embedding index |
| `TensorRegex` | 64 bytes (pattern code) | 64 bytes (match positions) | Pattern match over token sequence |
| `SafeEval` | 128 bytes (expression) | 64 bytes (numeric result + status) | Sandboxed arithmetic evaluation |

## Adding a New Capability — Checklist

1. **types.rs**: Add `NewCap = N` variant to `CapabilityId` (N = next integer).
2. **types.rs**: Define `NewCapArgs` and `NewCapResult` structs:
   ```rust
   #[repr(C)]
   #[derive(Copy, Clone, Pod, Zeroable)]
   pub struct NewCapArgs {
       // fixed-size fields only
       pub input: [u8; 128],
       pub flags: u32,
       pub _pad: [u8; 4],  // explicit padding to reach 136 bytes
   }
   // Always add a size assertion:
   const _: () = assert!(core::mem::size_of::<NewCapArgs>() == 136);
   ```
3. **cortex.rs**: Add match arm:
   ```rust
   CapabilityId::NewCap => {
       let args = bytemuck::from_bytes::<NewCapArgs>(args_bytes);
       let result = handle_new_cap(args);
       bytemuck::bytes_of(&result)
   }
   ```
4. **fabric.rs / types.rs**: Add `new_cap_obs: [u8; SIZE_OF_RESULT]` slice to `FabricLayout` observation buffers region.
5. Run `cargo check --workspace` to verify.

## Observation Buffer Sizing

Each capability gets a fixed-size slice in the Fabric for its most recent result. The Weaver reads these slices to build the next input context.

```
FabricLayout::observation_buffers:
  [0..512]   → CargoCheck result
  [512..560] → GitStatus result
  [560..816] → VectorSearch result
  [816..880] → TensorRegex result
  [880..944] → SafeEval result
  // New capabilities append here
```

Keep result sizes small — these live in the hot path of the inference loop. Prefer indices/codes over raw text.

## Common Mistakes

| Mistake | Consequence |
|---------|-------------|
| Padding in Args/Result struct | `bytemuck::from_bytes` panics (size mismatch) |
| Reusing a retired `CapabilityId` | Dispatches to wrong handler on old .aether files |
| Heap allocation in handler | Violates zero-copy contract; causes latency spikes |
| Panicking handler | Crashes inference loop — always return error codes |
| Forgetting size assertion | Silent size mismatch discovered only at runtime |
