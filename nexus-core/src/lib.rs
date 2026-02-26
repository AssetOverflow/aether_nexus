//! AetherNexus – nexus-core
//!
//! A Sovereign Tensor Organism for Apple Silicon.
//!
//! This crate implements the core architecture described in the Yellowpaper v1.3:
//! - Unified Fabric (single mmap'd tensor region)
//! - Weaver decode kernel (decompression-free sparse attention)
//! - Unified Capability Cortex (typed, zero-copy dispatch)
//! - ANE Distiller (background memory consolidation)
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │             .aether File (mmap)              │
//! │  ┌─────────┬──────────┬──────────┬────────┐  │
//! │  │ Weights │ Hot Pool │ Cold Pool│  Dict  │  │
//! │  │  (f16)  │  (f16)   │(Sparse)  │ (f16)  │  │
//! │  └────┬────┴────┬─────┴────┬─────┴───┬────┘  │
//! │       │         │          │         │        │
//! │  ┌────▼─────────▼──────────▼─────────▼────┐  │
//! │  │          Unified Fabric                │  │
//! │  │    (zero-copy sub-views via MLX)        │  │
//! │  └────────────────┬───────────────────────┘  │
//! │                   │                          │
//! │  ┌────────────────▼───────────────────────┐  │
//! │  │         Weaver Decode (GPU)            │  │
//! │  │   Hot path: exact f16 attention        │  │
//! │  │   Cold path: decompression-free O(4)   │  │
//! │  └────────────────┬───────────────────────┘  │
//! │                   │ action tensor            │
//! │  ┌────────────────▼───────────────────────┐  │
//! │  │      Capability Cortex (CPU)           │  │
//! │  │   CargoCheck │ VectorSearch │ Git │ ... │  │
//! │  └────────────────┬───────────────────────┘  │
//! │                   │ observation              │
//! │  ┌────────────────▼───────────────────────┐  │
//! │  │       ANE Distiller (background)       │  │
//! │  │   Entropy eval → Dict projection →     │  │
//! │  │   SparseCode packing                   │  │
//! │  └────────────────────────────────────────┘  │
//! └──────────────────────────────────────────────┘
//! ```

pub mod types;
pub mod fabric;
pub mod capability;
pub mod cortex;
pub mod distiller;
pub mod weaver;
pub mod bench;
pub mod weight_loader;
pub mod tokenizer;
pub mod ops;
pub mod inference;
pub mod agent;

// ─────────────────────────────────────────────────────────────────────────────
// Public re-exports
// ─────────────────────────────────────────────────────────────────────────────

pub use types::{
    // Model geometry
    ModelDims,
    Llama8B,
    Granite2B,
    // Memory types
    SparseCode,
    LoomDescriptor,
    Persona,
    // Capability system
    CapabilityId,
    CortexError,
    // Fabric
    FabricError,
    FabricLayout,
    WeaverParams,
    // Constants
    HEADER_MAGIC,
    FORMAT_VERSION,
    SIGNATURE_LEN,
    MAX_RESULT_SIZE,
    WAL_INTERVAL_MS,
    REM_INTERVAL_SECS,
    DISTILL_ENTROPY_THRESHOLD,
};

pub use fabric::{Fabric, FabricRegions, create_genesis};
pub use agent::AgentLoop;
