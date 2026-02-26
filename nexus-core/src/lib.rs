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

pub mod agent;
pub mod bench;
pub mod capability;
pub mod cortex;
pub mod distiller;
pub mod fabric;
pub mod inference;
pub mod ops;
pub mod sandbox;
pub mod tokenizer;
pub mod types;
pub mod weaver;
pub mod weight_loader;

// ─────────────────────────────────────────────────────────────────────────────
// Public re-exports
// ─────────────────────────────────────────────────────────────────────────────

pub use types::{
    CHECKPOINT_INTERVAL_MS,
    // Capability system
    CapabilityId,
    CortexError,
    DISTILL_ENTROPY_THRESHOLD,
    DeepSeekR1_1_5B,
    FORMAT_VERSION,
    // Fabric
    FabricError,
    FabricLayout,
    Granite2B,
    // Constants
    HEADER_MAGIC,
    Llama8B,
    LoomDescriptor,
    MAX_RESULT_SIZE,
    // Model geometry
    ModelDims,
    Persona,
    Qwen05B,
    REM_INTERVAL_SECS,
    SIGNATURE_LEN,
    // Configuration
    SecurityConfig,
    // Memory types
    SparseCode,
    WeaverParams,
};

pub use agent::AgentLoop;
pub use fabric::{Fabric, FabricRegions, create_genesis};
