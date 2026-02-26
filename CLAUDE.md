# AetherNexus — CLAUDE.md

## Project Overview

AetherNexus is a sovereign, self-contained AI inference engine written in Rust, optimized for Apple Silicon (M1+). It runs entirely air-gapped using Metal GPU compute via Apple's MLX framework. The `.aether` file is the single mmap'd state containing model weights, KV cache, persona looms, and cryptographic signatures.

## Architecture

### Workspace Layout

```
aether_nexus/
├── Cargo.toml                  # Workspace root (edition 2024, AGPL-3.0)
├── nexus-core/                 # Single crate
│   ├── build.rs                # Compiles Metal shaders (MSL → AIR → metallib)
│   └── src/
│       ├── main.rs             # Entry point / ignition sequence
│       ├── lib.rs              # Public API re-exports
│       ├── types.rs            # ModelDims, SparseCode, FabricLayout, CapabilityId
│       ├── fabric.rs           # Unified Fabric (mmap, WAL, Ed25519 signatures)
│       ├── capability.rs       # Capability trait + dispatch macro
│       ├── cortex.rs           # Capability Cortex dispatcher
│       ├── distiller.rs        # ANE Distiller (background REM cycle)
│       ├── weaver.rs           # Weaver Engine (Metal GPU decode)
│       ├── ops.rs              # GPU ops engine (matmul, RMSNorm, attention, FFN)
│       ├── inference.rs        # Autoregressive decode + KV cache
│       ├── weight_loader.rs    # Safetensors/GGUF weight loading
│       ├── tokenizer.rs        # HuggingFace tokenizer wrapper
│       ├── bench.rs            # GPU benchmark utilities
│       ├── weaver_kernel.metal # Sparse attention decode kernel (MSL)
│       └── ops.metal           # Transformer op kernels (MSL)
├── docs/                       # Whitepapers, yellowpaper, blueprints, phase walkthroughs
├── models/inference/           # Model weights (granite-2b-instruct)
└── cache/                      # HuggingFace model cache
```

### Core Components

| Component | File | Role |
|-----------|------|------|
| Unified Fabric | `fabric.rs` | Single mmap region: weights + KV pools + looms + WAL. Ed25519-signed. Flushes every 300 ms. |
| Cortex | `cortex.rs` | Type-safe dispatch of action tensors → native capabilities (CargoCheck, GitStatus, VectorSearch, TensorRegex, SafeEval) |
| Weaver Engine | `weaver.rs` + `weaver_kernel.metal` | GPU decode. Hot path: exact f16. Cold path: decompression-free O(4) sparse attention. |
| ANE Distiller | `distiller.rs` | Background REM cycle. Evaluates entropy, packs hot blocks → sparse codes, migrates to cold pool. |
| Ops Engine | `ops.rs` + `ops.metal` | GPU transformer ops: matmul, RMSNorm, masked attention, FFN. |
| Inference Engine | `inference.rs` | Autoregressive decode loop, KV cache management, token sampling. |

### Key Types (`types.rs`)

- `SparseCode` — 16 bytes (`[u16; 4]` indices + `[f16; 4]` coefficients). Single 128-bit M1 load.
- `FabricLayout` — Memory map layout descriptor.
- `CapabilityId` — Exhaustive enum preventing invalid Cortex dispatch.
- `ModelDims` — Trait parameterizing model architecture (layers, heads, hidden size, vocab).

## Build System

### Prerequisites

- Rust 1.85+ (edition 2024)
- Xcode Command Line Tools (required for Metal shader compilation)
- Apple Silicon Mac (M1+)

### Metal Shader Pipeline (`build.rs`)

```
.metal → xcrun metal -std=metal3.0 -O2 → .air → xcrun metallib → .metallib
```

Exports `WEAVER_METALLIB` and `OPS_METALLIB` env vars for runtime loading.

### Common Commands

```bash
cargo build --workspace               # Full build (compiles Metal shaders)
cargo run -p nexus-core               # Run with default brain.aether
cargo run -p nexus-core -- /path/to/brain.aether
cargo run -p nexus-core -- --bench    # GPU benchmark
cargo run -p nexus-core -- --generate "your prompt"
cargo test --workspace
```

## Models

| Model | Purpose | Location |
|-------|---------|----------|
| IBM Granite 3.0 2B Instruct | Primary inference (40L, 2048H, 32Q/8KV heads) | `cache/models--ibm-granite--granite-3.0-2b-instruct/` |
| Qwen3 Embedding 0.6B | Vector search (VectorSearch capability) | `cache/models--Qwen--Qwen3-Embedding-0.6B/` |
| Llama-3.1-8B | Default/fallback (32L, 32Q/8KV heads, 128H dim) | `models/inference/` |

## Design Constraints

- **Zero external network calls** — fully air-gapped at runtime.
- **Zero-copy semantics** — single mmap region shared by all components; no IPC or serialization overhead.
- **Compile-time correctness** — illegal states are unrepresentable (exhaustive enums, `Pod`/`Zeroable` bounds, borrow checker).
- **Capability args/results must implement `Pod + Zeroable`** — required for zero-copy Cortex dispatch.
- **All Fabric mutations go through WAL** — max data loss: 300 ms.
- **Ed25519 signature verified on every boot** — prevents tampered `.aether` files from loading.
- **No unsafe Fabric mutations outside designated observation buffers** — borrow checker enforces this.

## Performance Targets

| Metric | Target |
|--------|--------|
| Sustained throughput | 92–118 tokens/sec (128k context, 70% cold blocks) |
| Agent cycle latency | 35–65 ms (planner → coder → cargo_check → observe) |
| Cold start | <800 ms (mmap + signature verify) |
| Power | 0.55–0.95 W (P-cores sleep >99%) |

## Adding a New Capability

1. Add a variant to `CapabilityId` in `types.rs`.
2. Define `Args` and `Result` structs implementing `Pod + Zeroable + Capability`.
3. Implement the handler in `cortex.rs` (match arm in the dispatch loop).
4. Pre-allocate observation buffer slice in `FabricLayout`.

## Key Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `mlx-rs` | 0.21 | Apple MLX (Metal backend) |
| `metal` | 0.30 | Metal GPU programming |
| `bytemuck` | 1.19 | Pod/Zeroable zero-copy |
| `half` | 2.4 | f16 support |
| `memmap2` | 0.9 | Memory-mapped file I/O |
| `ring` | 0.17 | Ed25519 cryptographic signatures |
| `tokio` | 1.x | Async runtime (Distiller background task) |
| `tokenizers` | 0.22 | HuggingFace tokenizer |
| `safetensors` | 0.7 | Model weight format |
| `tree-sitter` | 0.25 | Syntax-aware capabilities |

## Documentation

- `docs/WHITEPAPER.md` — High-level architecture and performance claims.
- `docs/YELLOWPAPER.md` — Formal specification: type proofs, Metal kernel pseudocode, invariants.
- `docs/PROJECT_BLUEPRINTS_FORGED.md` — Distiller design and genesis bundle generation.
- `docs/REPOSITORY_SKELETON.md` — Full production-ready code skeleton.

## Claude Code Development Tooling

These live in `.claude/` and are invisible to `cargo build`. They give Claude Code project-specific context and automation during development.

### Slash Commands

| Command | Description |
|---------|-------------|
| `/check` | Run `cargo check` + `cargo clippy` and summarize all errors/warnings with fix suggestions |
| `/bench` | Run `--bench` and compare results against the 92-118 tok/s / 35-65 ms / <800ms targets |
| `/new-capability <Name> [description]` | Scaffold a full new Cortex capability (types, handler, FabricLayout slot) |
| `/distiller-status` | Audit Distiller REM config, pool layout, SparseCode format, and WAL interval |

### Subagents (auto-triggered by context)

| Agent | Triggers on | Color |
|-------|-------------|-------|
| `metal-kernel-reviewer` | Reviewing/debugging `.metal` files, GPU performance questions, threadgroup sizing | cyan |
| `rust-safety-auditor` | `unsafe` blocks, `Pod`/`Zeroable` derivations, `fabric.rs` mutations, async races | red |
| `capability-designer` | Adding/debugging Cortex capabilities, `cortex.rs` handlers, `CapabilityId` variants | green |

### Guardrails (permanently denied — no exceptions)

Enforced by `.claude/hooks/pre-tool-guard.sh` via a `PreToolUse` Bash hook. Hard blocks — cannot be overridden by any prompt.

| Category | Blocked pattern | Reason |
|----------|----------------|--------|
| Git | `git push --force / -f` | Prevents overwriting shared history |
| Git | `git reset --hard` | Destroys uncommitted work |
| Git | `git clean -f` | Permanently deletes untracked files |
| Git | `git branch -D` | Force-deletes branches without safety check |
| Files | `rm -rf` | Recursive force-delete of any path |
| Files | `rm/mv/truncate/dd` on `*.aether` | Protects signed Fabric state — irreplaceable without genesis |
| Files | Write to `~/.ssh`, `/etc`, `/usr`, `/System`, `~/.gnupg` | System/credential directory protection |
| Network | `curl`, `wget`, `fetch` | Air-gapped runtime principle |
| Cargo | `cargo publish` | Releases must be performed manually by the user |

Hook: `.claude/hooks/pre-tool-guard.sh` — registered in `.claude/settings.json`.

### Skills (auto-loaded by topic)

| Skill | Activates when discussing |
|-------|--------------------------|
| `metal-shader-dev` | MSL code, buffer bindings, threadgroup memory, GQA broadcasting, build.rs shader pipeline |
| `aether-fabric-ops` | Fabric layout, WAL flush order, hot/cold KV pools, Ed25519 signatures, mmap semantics |
| `capability-system` | Cortex dispatch pipeline, `Capability` trait, `CapabilityId`, observation buffer sizing |
