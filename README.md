# AetherNexus

**A Sovereign Tensor Organism for Apple Silicon**

AetherNexus is a single, persistent, self-verifying computational entity that collapses inference, memory, orchestration, and tool execution into one mmap'd `.aether` file and one Rust binary.

## Architecture

```
┌──────────────────────────────────────────────────┐
│             .aether File (mmap)                  │
│  ┌─────────┬──────────┬──────────┬────────┐      │
│  │ Weights │ Hot Pool │ Cold Pool│  Dict  │      │
│  │  (f16)  │  (f16)   │(Sparse)  │ (f16)  │      │
│  └────┬────┴────┬─────┴────┬─────┴───┬────┘      │
│       │         │          │         │            │
│  ┌────▼─────────▼──────────▼─────────▼────┐      │
│  │          Unified Fabric                │      │
│  │    (zero-copy sub-views via MLX)       │      │
│  └────────────────┬───────────────────────┘      │
│                   │                              │
│  ┌────────────────▼───────────────────────┐      │
│  │         Weaver Decode (GPU)            │      │
│  │   Hot path: exact f16 attention        │      │
│  │   Cold path: decompression-free O(4)   │      │
│  └────────────────┬───────────────────────┘      │
│                   │ action tensor                │
│  ┌────────────────▼───────────────────────┐      │
│  │      Capability Cortex (CPU)           │      │
│  │   CargoCheck │ VectorSearch │ Git │ ...│      │
│  └────────────────┬───────────────────────┘      │
│                   │ observation                  │
│  ┌────────────────▼───────────────────────┐      │
│  │       ANE Distiller (background)       │      │
│  │   Entropy eval → Dict projection →     │      │
│  │   SparseCode packing                   │      │
│  └────────────────────────────────────────┘      │
└──────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Build
cargo build --workspace

# Run (auto-creates genesis .aether if needed)
cargo run -p nexus-core

# Run with specific brain file
cargo run -p nexus-core -- /path/to/brain.aether

# Test all invariants
cargo test --workspace
```

## Requirements

- **Rust 1.85+** (edition 2024)
- **macOS** with Xcode Command Line Tools (for Metal shader compilation)
- **Apple Silicon** (M1 or later recommended)

## Project Structure

```
aether_nexus/
├── Cargo.toml              # Workspace root
├── nexus-core/
│   ├── Cargo.toml          # Core crate dependencies
│   ├── build.rs            # Metal shader compilation pipeline
│   └── src/
│       ├── main.rs         # Ignition sequence
│       ├── lib.rs          # Public re-exports
│       ├── types.rs        # Core types (ModelDims, SparseCode, etc.)
│       ├── fabric.rs       # Unified Fabric (mmap, WAL, Ed25519)
│       ├── capability.rs   # Capability trait & macros
│       ├── cortex.rs       # Unified Capability Cortex
│       ├── distiller.rs    # ANE Distiller (REM cycle)
│       └── weaver_kernel.metal  # GPU decode kernel
├── scripts/
│   └── bundle.rs           # Offline .aether genesis generator
├── docs/
│   ├── WHITEPAPER.md       # Visionary manifesto
│   ├── YELLOWPAPER.md      # Formal specification
│   ├── PROJECT_BLUEPRINTS_FORGED.md
│   └── REPOSITORY_SKELETON.md
└── README.md
```

## Yellowpaper Invariants (Compiler-Enforced)

| Invariant | Enforcement |
|-----------|-------------|
| SparseCode = 16 bytes | `#[repr(C, packed)]` + unit test |
| Fabric = 16 KB aligned | `align_up()` + unit test |
| Zero-copy dispatch | `split_at_mut` + borrow checker |
| Typed capabilities | `Pod + Zeroable` + exhaustive enum |
| Cryptographic genesis | Ed25519 (ring) sign + verify |
| WAL atomicity | `mmap.flush_async()` every 300ms |

## Documentation

- [Whitepaper v1.2](docs/WHITEPAPER.md) – The visionary manifesto
- [Yellowpaper v1.3](docs/YELLOWPAPER.md) – Formal specification
- [Repository Skeleton](docs/REPOSITORY_SKELETON.md) – Original design
- [Blueprints](docs/PROJECT_BLUEPRINTS_FORGED.md) – Distiller & bundle design

---

**The Fabric holds. Forge eternal.**
