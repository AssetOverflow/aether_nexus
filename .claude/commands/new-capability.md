---
description: Scaffold a new Cortex capability — generates types, handler, and FabricLayout slot
argument-hint: <CapabilityName> [description]
allowed-tools: Read, Edit, Grep
---

Use the capability-designer agent to scaffold a new Cortex capability named `$1`.

Description / purpose: $ARGUMENTS

The agent should:

1. Read `nexus-core/src/types.rs` to find the highest existing `CapabilityId` discriminant and the current observation buffer layout.
2. Read `nexus-core/src/capability.rs` to confirm the `Capability` trait signature.
3. Read `nexus-core/src/cortex.rs` to see existing handler patterns.
4. Generate and apply all required changes:
   - New `CapabilityId::$1 = N` variant in `types.rs`
   - `$1Args` and `$1Result` structs with `#[repr(C)]`, `Pod`, `Zeroable`, and `const` size assertions
   - Handler function `handle_$1(args: &$1Args) -> $1Result` in `cortex.rs`
   - Match arm wired into the Cortex dispatch loop
   - New observation buffer slot appended to `FabricLayout` with its byte range documented in a comment
5. Confirm with `cargo check --workspace` that the scaffold compiles cleanly.
6. Print a summary: CapabilityId discriminant, Args size (bytes), Result size (bytes), observation buffer offset range.
