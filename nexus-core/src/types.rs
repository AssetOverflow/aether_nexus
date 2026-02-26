//! Core types for AetherNexus – Yellowpaper v1.3 binding
//!
//! Every type in this module enforces the Yellowpaper's memory geometry invariants
//! at compile time. The Rust compiler and `bytemuck` guarantees ensure:
//!
//! - `SparseCode` is exactly 16 bytes (single 128-bit vector load on M1)
//! - `Fabric` is 16 KB page-aligned (contiguous DRAM mapping guarantee)
//! - All capability arguments and results are `Pod + Zeroable`

use bytemuck::{Pod, Zeroable};
use half::f16;
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// System Configuration (nexus.toml)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NexusConfig {
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_reflection_steps: usize,
    pub max_tokens: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_reflection_steps: 10,
            max_tokens: 128,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub distill_entropy_threshold: f32,
    pub rem_interval_secs: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            distill_entropy_threshold: 0.15,
            rem_interval_secs: 30,
        }
    }
}

impl Default for NexusConfig {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            memory: MemoryConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

/// Security configuration for the sandbox policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Allowed workspace root directories. File operations are restricted to these.
    #[serde(default)]
    pub workspace_roots: Vec<String>,
    /// Allowed subprocess binary names (e.g. "cargo", "git").
    #[serde(default = "SecurityConfig::default_commands")]
    pub allowed_commands: Vec<String>,
    /// Maximum subprocess execution timeout in milliseconds.
    #[serde(default = "SecurityConfig::default_timeout")]
    pub max_subprocess_timeout_ms: u64,
    /// Wasmtime fuel limit per WASM execution.
    #[serde(default = "SecurityConfig::default_fuel")]
    pub wasm_fuel: u64,
    /// Wasmtime memory page limit (64 KB per page).
    #[serde(default = "SecurityConfig::default_memory_pages")]
    pub wasm_memory_pages: u32,
}

impl SecurityConfig {
    fn default_commands() -> Vec<String> {
        vec![
            "cargo".into(), "git".into(), "ls".into(), "cat".into(),
            "head".into(), "tail".into(), "find".into(), "grep".into(),
            "wc".into(), "echo".into(), "mkdir".into(), "rustfmt".into(),
        ]
    }
    fn default_timeout() -> u64 { 30_000 }
    fn default_fuel() -> u64 { 1_000_000 }
    fn default_memory_pages() -> u32 { 256 }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            workspace_roots: Vec::new(),
            allowed_commands: Self::default_commands(),
            max_subprocess_timeout_ms: Self::default_timeout(),
            wasm_fuel: Self::default_fuel(),
            wasm_memory_pages: Self::default_memory_pages(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model Dimensions (Dual-System Architecture)
//
// System 1 (fast reflex):  Qwen 2.5 0.5B Instruct
// System 2 (deep reason):  DeepSeek-R1-Distill-Qwen 1.5B
// Legacy:                  Granite 3.0 2B, Llama 3.1 8B
// ─────────────────────────────────────────────────────────────────────────────

/// Trait defining the model's tensor dimensions.
///
/// Default values correspond to Llama-3.1-8B architecture (used for
/// Fabric layout sizing when no specific model is specified). The actual
/// inference models are configured via `InferenceConfig::detect_from_dir()`.
///
/// Override for different model geometries.
pub trait ModelDims: Send + Sync + 'static {
    const LAYERS: usize = 32;
    const Q_HEADS: usize = 32;
    const KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const HIDDEN_SIZE: usize = Self::Q_HEADS * Self::HEAD_DIM;
    const INTERMEDIATE_SIZE: usize = 14336;
    const VOCAB_SIZE: usize = 128256;
    const BLOCK_SIZE: usize = 16;       // tokens per macro-block
    const SPARSITY_K: usize = 4;        // top-4 sparse codes per token
    const DICT_SIZE: usize = 512;       // dictionary vectors per KV head

    /// GQA group size: how many Q heads share one KV head
    const GQA_GROUP: usize = Self::Q_HEADS / Self::KV_HEADS;
}

/// Llama-3.1-8B concrete dimensions (used for default Fabric layout)
pub struct Llama8B;
impl ModelDims for Llama8B {}

/// Granite 3.0 2B Instruct concrete dimensions
pub struct Granite2B;
impl ModelDims for Granite2B {
    const LAYERS: usize = 40;
    const Q_HEADS: usize = 32;
    const KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 64;     // 2048 / 32 = 64
    const HIDDEN_SIZE: usize = 2048;
    const INTERMEDIATE_SIZE: usize = 8192;
    const VOCAB_SIZE: usize = 49155;
}

/// Qwen 2.5 0.5B Instruct concrete dimensions (System 1 – fast reflex)
pub struct Qwen05B;
impl ModelDims for Qwen05B {
    const LAYERS: usize = 24;
    const Q_HEADS: usize = 14;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;     // 896 / 14 = 64
    const HIDDEN_SIZE: usize = 896;
    const INTERMEDIATE_SIZE: usize = 4864;
    const VOCAB_SIZE: usize = 151936;
}

/// DeepSeek-R1-Distill-Qwen 1.5B concrete dimensions (System 2 – deep reasoning)
pub struct DeepSeekR1_1_5B;
impl ModelDims for DeepSeekR1_1_5B {
    const LAYERS: usize = 28;
    const Q_HEADS: usize = 12;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;    // 1536 / 12 = 128
    const HIDDEN_SIZE: usize = 1536;
    const INTERMEDIATE_SIZE: usize = 8960;
    const VOCAB_SIZE: usize = 151936;
}

// ─────────────────────────────────────────────────────────────────────────────
// SparseCode – Atomic Unit of Cold Memory
// ─────────────────────────────────────────────────────────────────────────────

/// A sparse representation of a single token's KV contribution for one head.
///
/// Exactly 16 bytes → single 128-bit vector load on the M1 memory controller.
/// 4 dictionary indices (0..511) + 4 learned coefficients.
///
/// # Invariants
/// - `#[repr(C, align(16))]` ensures no padding → exactly `4*2 + 4*2 = 16` bytes
/// - `Pod + Zeroable` enables zero-copy reinterpretation from mmap'd memory
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SparseCode {
    /// Indices of the 4 dictionary vectors (u16 for 64k vocab)
    pub indices: [u16; 4],
    /// Coefficients for each vector (f16)
    pub coeffs: [f16; 4],
}

const _: () = assert!(std::mem::size_of::<SparseCode>() == 16);
const _: () = assert!(std::mem::align_of::<SparseCode>() == 16);



impl SparseCode {
    /// Create a zeroed SparseCode (all indices 0, all coefficients 0.0)
    pub const fn zero() -> Self {
        Self {
            indices: [0u16; 4],
            coeffs: [f16::ZERO; 4],
        }
    }
}

impl Default for SparseCode {
    fn default() -> Self {
        Self::zero()
    }
}



// ─────────────────────────────────────────────────────────────────────────────
// Capability IDs – Exhaustive, typed dispatch
// ─────────────────────────────────────────────────────────────────────────────

/// Exhaustive enumeration of all Cortex capabilities.
///
/// The verifier kernel uses this enum to reject illegal capability requests
/// at decode time. Adding a new capability requires:
/// 1. A new variant here
/// 2. A `Capability` trait implementation
/// 3. Registration in `Cortex::boot()`
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::EnumCount, strum::FromRepr)]
pub enum CapabilityId {
    /// Invoke `cargo check` on a workspace (via subprocess)
    CargoCheck = 0,
    /// Semantic vector search via MLX matmul
    VectorSearch = 1,
    /// Query git repository status (via libgit2)
    GitStatus = 2,
    /// Regex-powered text search (via grep-* crates)
    TensorRegex = 3,
    /// Restricted expression evaluation (via tree-sitter)
    SafeEval = 4,
    /// Read file from workspace
    FileRead = 5,
    /// Write file to workspace
    FileWrite = 6,
    /// List directory contents
    DirList = 7,
    /// Execute shell command
    ShellRunner = 8,
}

impl CapabilityId {
    pub const COUNT: usize = <Self as strum::EnumCount>::COUNT;

    /// Convert from raw u32 action tensor value
    pub fn from_raw(value: u32) -> Option<Self> {
        Self::from_repr(value)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loom Descriptor – Per-Persona Pathways
// ─────────────────────────────────────────────────────────────────────────────

/// Persona identifiers for the 4 cognitive agents within the organism.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Persona {
    Planner = 0,
    Coder = 1,
    Critic = 2,
    Reviewer = 3,
}

impl Persona {
    pub const COUNT: usize = 4;
}

/// A Loom descriptor defines a persona's attention pathway into the Fabric.
///
/// Contains 4 index tensors that determine which KV blocks each persona
/// attends to and how the hot/cold division is maintained.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LoomDescriptor {
    /// Which persona this Loom belongs to
    pub persona: Persona,
    /// Number of hot (exact f16) blocks referenced
    pub hot_count: u32,
    /// Number of cold (SparseCode) blocks referenced
    pub cold_count: u32,
    /// Offset into the Fabric's block reference array for this persona's hot refs
    pub hot_refs_offset: u32,
    /// Offset into the Fabric's block reference array for this persona's cold refs
    pub cold_refs_offset: u32,
    /// Current token position within this persona's context
    pub token_pos: u32,
    /// Padding for alignment
    pub _pad: [u32; 2],
}

impl Default for LoomDescriptor {
    fn default() -> Self {
        Self {
            persona: Persona::Planner,
            hot_count: 0,
            cold_count: 0,
            hot_refs_offset: 0,
            cold_refs_offset: 0,
            token_pos: 0,
            _pad: [0; 2],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weaver Kernel Parameters (Metal dispatch)
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters passed to the Weaver decode Metal kernel.
///
/// These are bound to `buffer(6)` in the MSL kernel as `constant WeaverParams&`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WeaverParams {
    /// Number of query heads
    pub q_heads: u32,
    /// Number of KV heads
    pub kv_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
    /// Tokens per macro-block
    pub block_size: u32,
    /// GQA group size (q_heads / kv_heads)
    pub gqa_group: u32,
    /// Number of hot blocks in current Loom
    pub hot_count: u32,
    /// Number of cold blocks in current Loom
    pub cold_count: u32,
    /// Dictionary size per KV head
    pub dict_size: u32,
    /// Sparsity K (top-K codes per token)
    pub sparsity_k: u32,
    /// Padding for 16-byte alignment
    pub _pad: [u32; 3],
}

// Safety: WeaverParams is repr(C), all fields are u32 (Copy + trivial).
unsafe impl Zeroable for WeaverParams {}
unsafe impl Pod for WeaverParams {}

impl WeaverParams {
    /// Construct WeaverParams from ModelDims and current Loom state
    pub fn from_loom<D: ModelDims>(loom: &LoomDescriptor) -> Self {
        Self {
            q_heads: D::Q_HEADS as u32,
            kv_heads: D::KV_HEADS as u32,
            head_dim: D::HEAD_DIM as u32,
            block_size: D::BLOCK_SIZE as u32,
            gqa_group: D::GQA_GROUP as u32,
            hot_count: loom.hot_count,
            cold_count: loom.cold_count,
            dict_size: D::DICT_SIZE as u32,
            sparsity_k: D::SPARSITY_K as u32,
            _pad: [0; 3],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cortex Error Type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising from Cortex capability dispatch.
#[derive(Debug, thiserror::Error)]
pub enum CortexError {
    #[error("invalid capability id: {0}")]
    InvalidId(u32),

    #[error("capability execution failed: {0}")]
    ExecutionFailed(String),

    #[error("argument slice out of bounds: offset={offset}, size={size}")]
    ArgOutOfBounds { offset: usize, size: usize },

    #[error("result slice out of bounds: offset={offset}, size={size}")]
    ResultOutOfBounds { offset: usize, size: usize },

    #[error("sandbox violation: {0}")]
    SandboxViolation(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

// ─────────────────────────────────────────────────────────────────────────────
// Fabric Error Type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising during Fabric operations (boot, persistence, mmap).
#[derive(Debug, thiserror::Error)]
pub enum FabricError {
    #[error("signature verification failed")]
    SignatureInvalid,

    #[error("invalid .aether file: {0}")]
    InvalidFile(String),

    #[error("mmap failed: {0}")]
    MmapFailed(#[from] std::io::Error),

    #[error("file too small: expected at least {expected} bytes, got {actual}")]
    FileTooSmall { expected: usize, actual: usize },

    #[error("invalid magic bytes")]
    BadMagic,

    #[error("version mismatch: expected {expected:#010x}, got {actual:#010x}")]
    VersionMismatch { expected: u32, actual: u32 },

    #[error("checkpoint flush failed: {0}")]
    CheckpointFailed(String),

    #[error("layout mismatch: {0}")]
    LayoutMismatch(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// File Format Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic bytes at the start of every `.aether` file
pub const HEADER_MAGIC: &[u8] = b"AETHER1.3";

/// Current file format version
pub const FORMAT_VERSION: u32 = 0x0001_0003;

/// Ed25519 signature length (bytes)
pub const SIGNATURE_LEN: usize = 64;

/// Maximum size of a Cortex result write (bytes)
pub const MAX_RESULT_SIZE: usize = 4096;

/// Checkpoint flush interval (milliseconds)
/// Checkpoint flush interval in milliseconds.
///
/// NOTE: This is NOT a write-ahead log. The "WAL" name was misleading.
/// This is a periodic mmap flush that does not provide atomicity.
pub const CHECKPOINT_INTERVAL_MS: u64 = 300;

/// Distiller REM cycle interval (seconds)
pub const REM_INTERVAL_SECS: u64 = 30;

/// Default entropy threshold for distillation candidacy
pub const DISTILL_ENTROPY_THRESHOLD: f32 = 0.15;

// ─────────────────────────────────────────────────────────────────────────────
// Fabric Layout Constants (computed from ModelDims)
// ─────────────────────────────────────────────────────────────────────────────

/// Fabric memory layout for a given model geometry.
///
/// All offsets are computed at const-eval time, ensuring the entire Fabric
/// layout is known at compile time.
pub struct FabricLayout;

impl FabricLayout {
    /// Header size: magic + version + signature
    pub const HEADER_SIZE: usize = HEADER_MAGIC.len() + 4 + SIGNATURE_LEN;

    /// Maximum hot blocks (tunable per deployment)
    pub const MAX_HOT_BLOCKS: usize = 8192;

    /// Maximum cold blocks
    pub const MAX_COLD_BLOCKS: usize = 32768;

    /// Observation buffer total size (bytes)
    pub const OBS_BUF_SIZE: usize = 256 * 1024 * 1024; // 256 MB

    /// Holographic trace log size (bytes)
    pub const TRACE_LOG_SIZE: usize = 128 * 1024 * 1024; // 128 MB
}

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time & runtime invariant assertions
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Yellowpaper §3: SparseCode MUST be exactly 16 bytes for single 128-bit load
    #[test]
    fn sparse_code_is_16_bytes() {
        assert_eq!(
            std::mem::size_of::<SparseCode>(),
            16,
            "SparseCode must be exactly 16 bytes for M1 128-bit vector load"
        );
    }

    /// Verify SparseCode alignment (repr(C, align(16)) → align 16)
    #[test]
    fn sparse_code_alignment() {
        assert_eq!(std::mem::align_of::<SparseCode>(), 16);
    }

    /// Verify default SparseCode is all zeroes
    #[test]
    fn sparse_code_default_is_zero() {
        let code = SparseCode::default();
        let bytes = bytemuck::bytes_of(&code);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    /// Verify Pod round-trip: bytes → SparseCode → bytes
    #[test]
    fn sparse_code_pod_round_trip() {
        let original = SparseCode {
            indices: [1, 42, 255, 511],
            coeffs: [
                f16::from_f32(0.5),
                f16::from_f32(-0.25),
                f16::from_f32(1.0),
                f16::from_f32(0.0),
            ],
        };
        let bytes = bytemuck::bytes_of(&original);
        assert_eq!(bytes.len(), 16);
        let recovered: &SparseCode = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.indices, original.indices);
    }

    /// WeaverParams must be Pod + Zeroable
    #[test]
    fn weaver_params_pod() {
        let params = WeaverParams {
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            block_size: 16,
            gqa_group: 4,
            hot_count: 100,
            cold_count: 500,
            dict_size: 512,
            sparsity_k: 4,
            _pad: [0; 3],
        };
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), std::mem::size_of::<WeaverParams>());
    }

    /// CapabilityId exhaustive coverage
    #[test]
    fn capability_id_from_raw_exhaustive() {
        for i in 0..CapabilityId::COUNT as u32 {
            assert!(CapabilityId::from_raw(i).is_some(), "Missing CapabilityId for {i}");
        }
        assert!(CapabilityId::from_raw(CapabilityId::COUNT as u32).is_none());
        assert!(CapabilityId::from_raw(u32::MAX).is_none());
    }

    /// ModelDims GQA group computation
    #[test]
    fn llama8b_gqa_group() {
        assert_eq!(Llama8B::GQA_GROUP, 4, "32 Q heads / 8 KV heads = 4");
    }

    /// Verify header magic length is reasonable
    #[test]
    fn header_magic_valid() {
        assert_eq!(HEADER_MAGIC, b"AETHER1.3");
        assert!(HEADER_MAGIC.len() < 64);
    }
}
