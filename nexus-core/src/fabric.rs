//! The Unified Fabric – Single source of truth for the organism's state.
//!
//! The Fabric is a single mmap'd `.aether` file that contains:
//! - Model weights (f16/Q4)
//! - Hot KV pool (exact f16 macro-blocks)
//! - Cold KV pool (SparseCode dictionary codes)
//! - Learned dictionary (per KV head)
//! - Loom descriptors (per-persona attention pathways)
//! - Radix metadata (shared-prefix deduplication)
//! - Observation buffers (Cortex zero-copy writes)
//! - Holographic trace (deterministic replay log)
//!
//! # Invariants (Yellowpaper v1.3)
//!
//! - The entire Fabric occupies a single contiguous mmap region
//! - All sub-regions are accessed via byte-offset slicing (zero-copy)
//! - Ed25519 signature verification on boot (opt-in via BootMode)
//! - Checkpoint flush every 300ms for persistence (periodic mmap flush,
//!   NOT a write-ahead log — no atomicity or crash-consistency guarantees)
//!
//! # Architecture
//!
//! The Fabric does NOT wrap `mlx_rs::Array` for its backing store.
//! Instead, it owns a raw `memmap2::Mmap` (or `MmapMut`) and creates
//! MLX Arrays on demand from sub-regions via `Array::from_raw_data`.
//! This preserves true zero-copy semantics on Apple Silicon UMA.

use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use memmap2::MmapMut;
use ring::signature;
use half::f16;

use crate::types::{
    FabricError, FabricLayout, LoomDescriptor, ModelDims, Persona, SparseCode,
    HEADER_MAGIC, FORMAT_VERSION, SIGNATURE_LEN, CHECKPOINT_INTERVAL_MS,
};

// ─────────────────────────────────────────────────────────────────────────────
// Boot Mode
// ─────────────────────────────────────────────────────────────────────────────

/// Controls whether signature verification is enforced at boot.
///
/// In development workflows, genesis keys are often ephemeral (generated fresh
/// each time), making verification impossible. This enum makes the decision
/// explicit rather than silently skipping verification.
#[derive(Debug, Clone)]
pub enum BootMode {
    /// Development mode: skip signature verification.
    /// Prints a visible warning at boot.
    Dev,
    /// Production mode: verify the `.aether` file signature against this
    /// public key. Fails hard with `FabricError::SignatureInvalid` if
    /// verification fails.
    Verified {
        /// The 32-byte Ed25519 public key to verify against.
        public_key: [u8; 32],
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Fabric Region Offsets
// ─────────────────────────────────────────────────────────────────────────────

/// Computed byte offsets for each region within the `.aether` file.
///
/// All offsets are deterministic given a `ModelDims` configuration.
/// This acts as the "address map" for the organism's memory.
#[derive(Debug, Clone)]
pub struct FabricRegions {
    /// Byte offset where model weights begin
    pub weights_offset: usize,
    /// Size of the weights region in bytes
    pub weights_size: usize,

    /// Byte offset where the hot KV pool begins
    pub hot_pool_offset: usize,
    /// Size of the hot pool in bytes
    pub hot_pool_size: usize,

    /// Byte offset where the cold KV pool begins
    pub cold_pool_offset: usize,
    /// Size of the cold pool in bytes
    pub cold_pool_size: usize,

    /// Byte offset where the dictionary begins
    pub dict_offset: usize,
    /// Size of the dictionary in bytes
    pub dict_size: usize,

    /// Byte offset where Loom descriptors begin
    pub looms_offset: usize,
    /// Size of the Loom region in bytes
    pub looms_size: usize,

    /// Byte offset where radix metadata begins
    pub radix_offset: usize,
    /// Size of the radix metadata in bytes
    pub radix_size: usize,

    /// Byte offset where observation buffers begin
    pub obs_offset: usize,
    /// Size of observation buffers in bytes
    pub obs_size: usize,

    /// Byte offset where holographic trace begins
    pub trace_offset: usize,
    /// Size of the trace log in bytes
    pub trace_size: usize,

    /// Total size of the Fabric in bytes
    pub total_size: usize,
}

impl FabricRegions {
    /// Compute region offsets for a given model geometry.
    ///
    /// Layout order:
    /// [Header | Weights | Hot Pool | Cold Pool | Dictionary | Looms | Radix | Obs Bufs | Trace | Signature]
    pub fn compute<D: ModelDims>() -> Self {
        let header_size = FabricLayout::HEADER_SIZE;

        // Weights: approximate allocation based on ModelDims
        let weights_offset = align_up(header_size, 16384);
        let weights_size = align_up(Self::estimated_weight_bytes::<D>(), 16384);

        // Hot KV Pool: [MAX_HOT_BLOCKS * BLOCK_SIZE * KV_HEADS * HEAD_DIM * 2 bytes (f16)]
        let hot_pool_offset = align_up(weights_offset + weights_size, 16384);
        let hot_pool_size = FabricLayout::MAX_HOT_BLOCKS
            * D::BLOCK_SIZE
            * D::KV_HEADS
            * D::HEAD_DIM
            * 2; // f16 = 2 bytes

        // Cold KV Pool: [MAX_COLD_BLOCKS * BLOCK_SIZE * KV_HEADS * sizeof(SparseCode)]
        let cold_pool_offset = align_up(hot_pool_offset + hot_pool_size, 16384);
        let cold_pool_size = FabricLayout::MAX_COLD_BLOCKS
            * D::BLOCK_SIZE
            * D::KV_HEADS
            * std::mem::size_of::<SparseCode>();

        // Dictionary: [KV_HEADS * DICT_SIZE * HEAD_DIM * 2 bytes (f16)]
        let dict_offset = align_up(cold_pool_offset + cold_pool_size, 16384);
        let dict_size = D::KV_HEADS * D::DICT_SIZE * D::HEAD_DIM * 2;

        // Loom descriptors: 4 personas
        let looms_offset = align_up(dict_offset + dict_size, 4096);
        let looms_size = Persona::COUNT * std::mem::size_of::<LoomDescriptor>();

        // Radix metadata: [MAX_BLOCKS * 3 * 4 bytes (u32)]
        let max_blocks = FabricLayout::MAX_HOT_BLOCKS + FabricLayout::MAX_COLD_BLOCKS;
        let radix_offset = align_up(looms_offset + looms_size, 4096);
        let radix_size = max_blocks * 3 * 4;

        // Observation buffers
        let obs_offset = align_up(radix_offset + radix_size, 16384);
        let obs_size = FabricLayout::OBS_BUF_SIZE;

        // Holographic trace
        let trace_offset = align_up(obs_offset + obs_size, 16384);
        let trace_size = FabricLayout::TRACE_LOG_SIZE;

        // Total (with signature at end)
        let total_size = align_up(trace_offset + trace_size + SIGNATURE_LEN, 16384);

        FabricRegions {
            weights_offset,
            weights_size,
            hot_pool_offset,
            hot_pool_size,
            cold_pool_offset,
            cold_pool_size,
            dict_offset,
            dict_size,
            looms_offset,
            looms_size,
            radix_offset,
            radix_size,
            obs_offset,
            obs_size,
            trace_offset,
            trace_size,
            total_size,
        }
    }

    /// Approximate weight storage: all projection matrices + embeddings + norms.
    /// formula: 2 * hidden * (4*hidden + 2*kv_ratio*hidden + intermediate + vocab + layers*norms)
    fn estimated_weight_bytes<D: ModelDims>() -> usize {
        let h = D::HIDDEN_SIZE;
        let inter = D::INTERMEDIATE_SIZE;
        let vocab = D::VOCAB_SIZE;
        let layers = D::LAYERS;
        let kv_h = D::KV_HEADS;
        let q_h = D::Q_HEADS;
        let hd = D::HEAD_DIM;
        
        let per_layer = 2 * (
            q_h * hd * h      // q_proj
            + kv_h * hd * h   // k_proj
            + kv_h * hd * h   // v_proj
            + q_h * hd * h    // o_proj
            + h * inter       // gate_proj
            + h * inter       // up_proj
            + inter * h       // down_proj
            + h               // input_layernorm
            + h               // post_attention_layernorm
        );
        let global = 2 * (vocab * h + h); // embed_tokens + final_norm
        layers * per_layer + global
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The Fabric
// ─────────────────────────────────────────────────────────────────────────────

/// The Unified Fabric – the organism's entire cognitive state in one mmap region.
///
/// This struct owns the mmap'd `.aether` file and provides zero-copy access
/// to all sub-regions. It is the single source of truth.
///
/// # Thread Safety
///
/// The Fabric uses interior mutability for observation buffers (written by
/// the Cortex during dispatch) and the trace log. All other regions are
/// read-only after boot.
pub struct Fabric<D: ModelDims> {
    /// The mmap'd .aether file (mutable for checkpoint + observation writes)
    mmap: MmapMut,

    /// Computed region offsets
    pub regions: FabricRegions,

    /// Parsed Loom descriptors (in-memory working copies)
    pub looms: [LoomDescriptor; 4],

    /// Whether the Fabric has been modified since last checkpoint flush
    dirty: AtomicBool,

    /// Last checkpoint flush timestamp
    last_checkpoint: std::sync::Mutex<Instant>,

    /// Type witness (zero-sized, erased at runtime)
    _dims: std::marker::PhantomData<D>,
}

impl<D: ModelDims> Fabric<D> {
    /// Boot the Fabric from a signed `.aether` file.
    ///
    /// This is the organism's ignition sequence:
    /// 1. Open and mmap the file
    /// 2. Verify magic bytes and version
    /// 3. Optionally verify Ed25519 signature (based on `BootMode`)
    /// 4. Compute region offsets
    /// 5. Parse Loom descriptors
    ///
    /// # Errors
    ///
    /// Returns `FabricError` if the file is invalid, too small, or fails
    /// signature verification (in `Verified` mode).
    pub fn boot(path: &str) -> Result<Self, FabricError> {
        Self::boot_with_mode(path, BootMode::Dev)
    }

    /// Boot the Fabric with explicit signature verification control.
    pub fn boot_with_mode(path: &str, mode: BootMode) -> Result<Self, FabricError> {
        let path = Path::new(path);

        // Open for read+write (checkpoint writes)
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;

        let regions = FabricRegions::compute::<D>();

        // Verify file size
        let file_len = file.metadata()?.len() as usize;
        if file_len < regions.total_size {
            return Err(FabricError::FileTooSmall {
                expected: regions.total_size,
                actual: file_len,
            });
        }

        // mmap the entire file
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Verify magic bytes
        if &mmap[..HEADER_MAGIC.len()] != HEADER_MAGIC {
            return Err(FabricError::BadMagic);
        }

        // Verify version
        let version_bytes: [u8; 4] = mmap[HEADER_MAGIC.len()..HEADER_MAGIC.len() + 4]
            .try_into()
            .map_err(|_| FabricError::InvalidFile("version bytes".into()))?;
        let version = u32::from_le_bytes(version_bytes);
        if version != FORMAT_VERSION {
            return Err(FabricError::VersionMismatch {
                expected: FORMAT_VERSION,
                actual: version,
            });
        }

        // Signature verification based on BootMode
        match &mode {
            BootMode::Dev => {
                eprintln!("[WARN] Booting in DEV mode — signature verification SKIPPED");
            }
            BootMode::Verified { public_key } => {
                let data_end = regions.total_size - SIGNATURE_LEN;
                let sig_start = data_end;
                let sig_end = data_end + SIGNATURE_LEN;

                let public_key = signature::UnparsedPublicKey::new(
                    &signature::ED25519,
                    public_key,
                );

                public_key
                    .verify(&mmap[..data_end], &mmap[sig_start..sig_end])
                    .map_err(|_| FabricError::SignatureInvalid)?;

                println!("[BOOT] Ed25519 signature verified ✓");
            }
        }

        // Parse Loom descriptors from the mmap
        let looms = Self::parse_looms(&mmap, &regions);

        Ok(Self {
            mmap,
            regions,
            looms,
            dirty: AtomicBool::new(false),
            last_checkpoint: std::sync::Mutex::new(Instant::now()),
            _dims: std::marker::PhantomData,
        })
    }

    // ─── Sub-region accessors (zero-copy) ────────────────────────────────

    /// Read-only view of the model weights region
    pub fn weights(&self) -> &[u8] {
        &self.mmap[self.regions.weights_offset..self.regions.weights_offset + self.regions.weights_size]
    }

    /// Read-only view of the model weights region as f16
    pub fn weight_region_as_f16(&self) -> &[f16] {
        let bytes = self.weights();
        bytemuck::cast_slice(bytes)
    }

    /// Read-only view of the hot KV pool
    pub fn hot_pool(&self) -> &[u8] {
        &self.mmap[self.regions.hot_pool_offset..self.regions.hot_pool_offset + self.regions.hot_pool_size]
    }

    /// Read-only view of the cold KV pool (SparseCode blocks)
    pub fn cold_pool(&self) -> &[u8] {
        &self.mmap[self.regions.cold_pool_offset..self.regions.cold_pool_offset + self.regions.cold_pool_size]
    }

    /// The cold pool reinterpreted as SparseCode slice
    pub fn cold_pool_codes(&self) -> &[SparseCode] {
        let bytes = self.cold_pool();
        bytemuck::cast_slice(bytes)
    }

    /// Read-only view of the dictionary
    pub fn dictionary(&self) -> &[u8] {
        &self.mmap[self.regions.dict_offset..self.regions.dict_offset + self.regions.dict_size]
    }

    /// Read-only view of the radix metadata
    pub fn radix_metadata(&self) -> &[u8] {
        &self.mmap[self.regions.radix_offset..self.regions.radix_offset + self.regions.radix_size]
    }

    // ─── Mutable accessors (for Cortex dispatch + WAL) ───────────────────

    /// Mutable view of the observation buffers for Cortex writes.
    ///
    /// # Safety Contract
    ///
    /// The caller MUST ensure that only one Cortex dispatch writes to any
    /// given sub-region at a time. The Cortex enforces this via its
    /// `arg_offset` / `result_offset` dispatch protocol.
    pub fn observation_bufs_mut(&mut self) -> &mut [u8] {
        let start = self.regions.obs_offset;
        let end = start + self.regions.obs_size;
        self.dirty.store(true, Ordering::Release);
        &mut self.mmap[start..end]
    }

    /// Read-only view of observation buffers
    pub fn observation_bufs(&self) -> &[u8] {
        &self.mmap[self.regions.obs_offset..self.regions.obs_offset + self.regions.obs_size]
    }

    /// Mutable view of the cold pool for distillation writes
    pub fn cold_pool_mut(&mut self) -> &mut [u8] {
        let start = self.regions.cold_pool_offset;
        let end = start + self.regions.cold_pool_size;
        self.dirty.store(true, Ordering::Release);
        &mut self.mmap[start..end]
    }

    /// Mutable view of the holographic trace log
    pub fn trace_log_mut(&mut self) -> &mut [u8] {
        let start = self.regions.trace_offset;
        let end = start + self.regions.trace_size;
        &mut self.mmap[start..end]
    }
    // ─── Holographic Trace ────────────────────────────────────────────────

    /// Append a structured trajectory record to the Holographic Trace ring buffer.
    pub fn append_trace(&mut self, record: &str) {
        let record_bytes = record.as_bytes();
        let len = record_bytes.len();
        let trace_size = self.regions.trace_size;
        
        if len == 0 || len > trace_size - 4 {
            return; // Too big or empty
        }

        let trace_buf = self.trace_log_mut();
        
        let mut cursor_bytes = [0u8; 4];
        cursor_bytes.copy_from_slice(&trace_buf[0..4]);
        let mut cursor = u32::from_le_bytes(cursor_bytes) as usize;
        
        if cursor < 4 || cursor + len > trace_size {
            cursor = 4; // Reset to beginning
        }
        
        trace_buf[cursor..cursor + len].copy_from_slice(record_bytes);
        cursor += len;
        
        trace_buf[0..4].copy_from_slice(&(cursor as u32).to_le_bytes());
        self.dirty.store(true, Ordering::Release);
    }

    // ─── Persistence (Checkpoint) ────────────────────────────────────────
    //
    // NOTE: This is a periodic mmap flush, NOT a write-ahead log.
    // It does not provide atomicity or crash-consistency guarantees.
    // A crash during flush may leave the .aether file in a partially
    // updated state. For true durability, a proper WAL or snapshot
    // mechanism would be needed.

    /// Flush dirty pages to disk if the checkpoint interval has elapsed.
    ///
    /// Called at the end of each cognitive cycle. Only flushes if:
    /// 1. The Fabric has been modified (dirty flag)
    /// 2. At least CHECKPOINT_INTERVAL_MS has elapsed since last flush
    pub fn checkpoint(&mut self) -> Result<(), FabricError> {
        if !self.dirty.load(Ordering::Acquire) {
            return Ok(());
        }

        let mut last = self.last_checkpoint.lock().unwrap();
        if last.elapsed() < Duration::from_millis(CHECKPOINT_INTERVAL_MS) {
            return Ok(());
        }

        // Flush the mmap to disk (async flush for performance)
        self.mmap
            .flush_async()
            .map_err(|e| FabricError::CheckpointFailed(e.to_string()))?;

        self.dirty.store(false, Ordering::Release);
        *last = Instant::now();

        Ok(())
    }

    /// Force an immediate synchronous flush (for shutdown)
    pub fn force_checkpoint(&mut self) -> Result<(), FabricError> {
        self.mmap
            .flush()
            .map_err(|e| FabricError::CheckpointFailed(e.to_string()))?;
        self.dirty.store(false, Ordering::Release);
        *self.last_checkpoint.lock().unwrap() = Instant::now();
        Ok(())
    }

    // ─── Signature verification ──────────────────────────────────────────

    /// Verify the Ed25519 signature of the `.aether` file.
    ///
    /// The signature covers all bytes from offset 0 to (total_size - SIGNATURE_LEN).
    /// The signature itself is stored in the last SIGNATURE_LEN bytes.
    pub fn verify_signature(&self, public_key_bytes: &[u8]) -> Result<(), FabricError> {
        let public_key = signature::UnparsedPublicKey::new(
            &signature::ED25519,
            public_key_bytes,
        );

        let data_end = self.regions.total_size - SIGNATURE_LEN;
        let sig_start = data_end;
        let sig_end = data_end + SIGNATURE_LEN;

        let data = &self.mmap[..data_end];
        let sig = &self.mmap[sig_start..sig_end];

        public_key
            .verify(data, sig)
            .map_err(|_| FabricError::SignatureInvalid)
    }

    // ─── Internal helpers ────────────────────────────────────────────────

    /// Parse LoomDescriptor structs from the mmap region
    fn parse_looms(mmap: &MmapMut, regions: &FabricRegions) -> [LoomDescriptor; 4] {
        let mut looms = [LoomDescriptor::default(); 4];
        let loom_bytes = &mmap[regions.looms_offset..regions.looms_offset + regions.looms_size];
        let loom_size = std::mem::size_of::<LoomDescriptor>();

        for (i, loom) in looms.iter_mut().enumerate() {
            let offset = i * loom_size;
            if offset + loom_size <= loom_bytes.len() {
                // Read individual fields to avoid packed struct issues
                let persona_byte = loom_bytes[offset];
                loom.persona = match persona_byte {
                    0 => Persona::Planner,
                    1 => Persona::Coder,
                    2 => Persona::Critic,
                    3 => Persona::Reviewer,
                    _ => Persona::Planner, // default fallback
                };

                // Read u32 fields (little-endian)
                let base = offset + 4; // skip persona byte + padding
                if base + 20 <= loom_bytes.len() {
                    loom.hot_count = read_u32_le(loom_bytes, base);
                    loom.cold_count = read_u32_le(loom_bytes, base + 4);
                    loom.hot_refs_offset = read_u32_le(loom_bytes, base + 8);
                    loom.cold_refs_offset = read_u32_le(loom_bytes, base + 12);
                    loom.token_pos = read_u32_le(loom_bytes, base + 16);
                }
            }
        }

        looms
    }

    /// Get the total size this Fabric occupies
    pub fn total_size(&self) -> usize {
        self.regions.total_size
    }

    /// Raw access to the full mmap (for advanced use)
    pub fn raw_bytes(&self) -> &[u8] {
        &self.mmap
    }

    // ─── Weight Embedding ────────────────────────────────────────────────
    //
    // The weights region stores model tensors in a self-describing format:
    //
    //   [4 bytes: WEIGHT_MAGIC "WGHT"]
    //   [4 bytes: num_layers (u32 LE)]
    //   [4 bytes: flags (u32 LE) — bit 0 = has_qkv_biases]
    //   [padding to 64-byte boundary]
    //   [tensor data: embed_tokens | layer[0..N] | final_norm | lm_head]
    //
    // Each layer's tensors are concatenated in fixed order:
    //   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
    //   input_layernorm, post_attention_layernorm, [q_bias, k_bias, v_bias]
    //
    // All tensors are f16 (2 bytes per element), stored contiguously.
    // Since tensor sizes are deterministic from InferenceConfig, no per-tensor
    // size headers are needed — offsets can be recomputed at load time.

    /// Magic bytes at the start of the weights region indicating weights are embedded.
    const WEIGHT_MAGIC: [u8; 4] = *b"WGHT";

    /// Header size for the weight manifest (aligned to 64 bytes).
    const WEIGHT_HEADER_SIZE: usize = 64;

    /// Check whether model weights have been embedded into this Fabric.
    pub fn weights_embedded(&self) -> bool {
        let w = self.weights();
        w.len() >= 4 && w[..4] == Self::WEIGHT_MAGIC
    }

    /// Read the weight manifest header. Returns (num_layers, has_biases).
    pub fn weight_manifest(&self) -> Option<(usize, bool)> {
        let w = self.weights();
        if w.len() < Self::WEIGHT_HEADER_SIZE || w[..4] != Self::WEIGHT_MAGIC {
            return None;
        }
        let num_layers = u32::from_le_bytes([w[4], w[5], w[6], w[7]]) as usize;
        let flags = u32::from_le_bytes([w[8], w[9], w[10], w[11]]);
        let has_biases = (flags & 1) != 0;
        Some((num_layers, has_biases))
    }

    /// Raw access to the weight data region (after the header).
    /// Returns the slice starting at the weight header boundary.
    pub fn weight_data(&self) -> &[u8] {
        let w = self.weights();
        if w.len() <= Self::WEIGHT_HEADER_SIZE {
            return &[];
        }
        &w[Self::WEIGHT_HEADER_SIZE..]
    }

    /// Embed model weights into the Fabric's weights region (mutable).
    ///
    /// This is called during bundling (genesis + weight embedding) to write
    /// all model tensors into the `.aether` file. After this, the `models/**`
    /// directory is no longer needed at runtime.
    ///
    /// # Arguments
    /// - `weight_bytes`: All tensors concatenated as raw f16 bytes in the
    ///   canonical order (embed, layers, norm, lm_head)
    /// - `num_layers`: Number of transformer layers
    /// - `has_biases`: Whether QKV biases are included per layer
    pub fn embed_weights(
        &mut self,
        weight_bytes: &[u8],
        num_layers: usize,
        has_biases: bool,
    ) -> Result<(), FabricError> {
        let header_plus_data = Self::WEIGHT_HEADER_SIZE + weight_bytes.len();
        if header_plus_data > self.regions.weights_size {
            return Err(FabricError::LayoutMismatch(format!(
                "Weight data ({} bytes) + header exceeds weights region ({} bytes)",
                weight_bytes.len(), self.regions.weights_size
            )));
        }

        let base = self.regions.weights_offset;

        // Write magic
        self.mmap[base..base + 4].copy_from_slice(&Self::WEIGHT_MAGIC);

        // Write num_layers
        self.mmap[base + 4..base + 8].copy_from_slice(&(num_layers as u32).to_le_bytes());

        // Write flags
        let flags: u32 = if has_biases { 1 } else { 0 };
        self.mmap[base + 8..base + 12].copy_from_slice(&flags.to_le_bytes());

        // Zero remaining header
        for b in &mut self.mmap[base + 12..base + Self::WEIGHT_HEADER_SIZE] {
            *b = 0;
        }

        // Write tensor data
        let data_start = base + Self::WEIGHT_HEADER_SIZE;
        self.mmap[data_start..data_start + weight_bytes.len()]
            .copy_from_slice(weight_bytes);

        self.dirty.store(true, Ordering::Release);
        println!("[FABRIC] Embedded {} bytes of weights ({} layers, biases={})",
            weight_bytes.len(), num_layers, has_biases);

        Ok(())
    }
}

/// Helper: read a u32 in little-endian from a byte slice
fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Align a value up to the next multiple of `alignment`.
const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Genesis: Create a new .aether file
// ─────────────────────────────────────────────────────────────────────────────

/// Create a fresh `.aether` genesis file with zeroed regions and Ed25519 signature.
///
/// This is used by the bundle script (`scripts/bundle.rs`) but is provided
/// as a library function for testing and programmatic creation.
pub fn create_genesis<D: ModelDims>(
    output_path: &str,
    signing_key: &signature::Ed25519KeyPair,
) -> Result<(), FabricError> {
    let regions = FabricRegions::compute::<D>();

    // Create the file and set its size
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_path)?;

    file.set_len(regions.total_size as u64)?;

    let mut mmap = unsafe { MmapMut::map_mut(&file)? };

    // Write header magic
    mmap[..HEADER_MAGIC.len()].copy_from_slice(HEADER_MAGIC);

    // Write version
    let version_offset = HEADER_MAGIC.len();
    mmap[version_offset..version_offset + 4].copy_from_slice(&FORMAT_VERSION.to_le_bytes());

    // Initialize Loom descriptors with default personas
    let loom_size = std::mem::size_of::<LoomDescriptor>();
    for i in 0..Persona::COUNT {
        let offset = regions.looms_offset + i * loom_size;
        mmap[offset] = i as u8; // persona byte
    }

    // All other regions are zeroed by the OS (mmap of new file)

    // Sign the content (everything except the signature itself)
    let data_end = regions.total_size - SIGNATURE_LEN;
    let sig = signing_key.sign(&mmap[..data_end]);
    let sig_bytes = sig.as_ref();
    mmap[data_end..data_end + sig_bytes.len()].copy_from_slice(sig_bytes);

    // Flush to disk
    mmap.flush().map_err(|e| FabricError::CheckpointFailed(e.to_string()))?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Llama8B;
    use ring::rand::SystemRandom;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use std::fs;

    /// Verify region offsets are 16 KB aligned where required
    #[test]
    fn regions_are_page_aligned() {
        let regions = FabricRegions::compute::<Llama8B>();

        assert_eq!(regions.weights_offset % 16384, 0, "weights must be 16KB aligned");
        assert_eq!(regions.hot_pool_offset % 16384, 0, "hot_pool must be 16KB aligned");
        assert_eq!(regions.cold_pool_offset % 16384, 0, "cold_pool must be 16KB aligned");
        assert_eq!(regions.dict_offset % 16384, 0, "dictionary must be 16KB aligned");
        assert_eq!(regions.obs_offset % 16384, 0, "observation bufs must be 16KB aligned");
        assert_eq!(regions.trace_offset % 16384, 0, "trace log must be 16KB aligned");
        assert_eq!(regions.total_size % 16384, 0, "total size must be 16KB aligned");
    }

    /// Verify regions don't overlap
    #[test]
    fn regions_non_overlapping() {
        let r = FabricRegions::compute::<Llama8B>();

        assert!(r.weights_offset + r.weights_size <= r.hot_pool_offset);
        assert!(r.hot_pool_offset + r.hot_pool_size <= r.cold_pool_offset);
        assert!(r.cold_pool_offset + r.cold_pool_size <= r.dict_offset);
        assert!(r.dict_offset + r.dict_size <= r.looms_offset);
        assert!(r.looms_offset + r.looms_size <= r.radix_offset);
        assert!(r.radix_offset + r.radix_size <= r.obs_offset);
        assert!(r.obs_offset + r.obs_size <= r.trace_offset);
    }

    /// Full genesis → boot → verify round-trip
    #[test]
    fn genesis_boot_verify_roundtrip() {
        let tmp_dir = std::env::temp_dir();
        let test_path = tmp_dir.join("test_genesis.aether");
        let test_path_str = test_path.to_str().unwrap();

        // Generate signing key
        let rng = SystemRandom::new();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();

        // Create genesis
        create_genesis::<Llama8B>(test_path_str, &key_pair).unwrap();

        // Boot from genesis
        let fabric = Fabric::<Llama8B>::boot(test_path_str).unwrap();

        // Verify signature
        let pub_key = key_pair.public_key().as_ref();
        fabric.verify_signature(pub_key).unwrap();

        // Verify Loom personas
        assert_eq!(fabric.looms[0].persona, Persona::Planner);
        assert_eq!(fabric.looms[1].persona, Persona::Coder);
        assert_eq!(fabric.looms[2].persona, Persona::Critic);
        assert_eq!(fabric.looms[3].persona, Persona::Reviewer);

        // Cleanup
        let _ = fs::remove_file(test_path);
    }

    /// Verify SparseCode zero-copy cast from cold pool
    #[test]
    fn cold_pool_sparse_code_cast() {
        let tmp_dir = std::env::temp_dir();
        let test_path = tmp_dir.join("test_sparse.aether");
        let test_path_str = test_path.to_str().unwrap();

        let rng = SystemRandom::new();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();

        create_genesis::<Llama8B>(test_path_str, &key_pair).unwrap();
        let fabric = Fabric::<Llama8B>::boot(test_path_str).unwrap();

        // Cold pool should be castable to SparseCode slice
        let codes = fabric.cold_pool_codes();
        assert!(!codes.is_empty());
        // All codes should be zeroed (genesis file)
        let first_indices = codes[0].indices;
        assert_eq!(first_indices, [0u16; 4]);

        let _ = fs::remove_file(test_path);
    }

    /// Verify align_up helper
    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 16384), 0);
        assert_eq!(align_up(1, 16384), 16384);
        assert_eq!(align_up(16384, 16384), 16384);
        assert_eq!(align_up(16385, 16384), 32768);
        assert_eq!(align_up(100, 4096), 4096);
    }
}
