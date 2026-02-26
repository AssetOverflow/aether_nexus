//! WeaverEngine – Rust-side Metal compute pipeline for the Weaver decode kernel.
//!
//! Orchestrates GPU dispatch of the Weaver kernel:
//! 1. Load the pre-compiled metallib (from build.rs)
//! 2. Create compute pipeline state
//! 3. Allocate GPU buffers for kernel inputs/outputs
//! 4. Encode and dispatch the kernel
//! 5. Read back the output token
//!
//! # Persistent Buffers
//!
//! All GPU input/output buffers are pre-allocated once at construction and reused
//! for every `decode()` call. Data is copied into existing buffers via the
//! shared-memory `contents()` pointer. Buffers are only reallocated if the
//! input exceeds the pre-allocated capacity (rare growth path).

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};

use crate::types::{SparseCode, WeaverParams};
use std::ffi::c_void;

/// Re-export half::f16 to avoid conflict with Rust 2024 native f16
use half::f16 as F16;

/// The Weaver Engine – GPU compute pipeline for the decode kernel.
///
/// Created once at boot, reused for every decode step.
/// Holds pre-allocated GPU buffers to avoid per-token buffer recreation.
pub struct WeaverEngine {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    _library: Library,

    // ── Persistent GPU buffers ─────────────────────────────────────────────
    // Pre-allocated once at construction, reused for every `decode()` call.
    // Reallocated only when input exceeds the pre-allocated capacity (rare path).
    buf_q_exact: Buffer,
    buf_q_latent: Buffer,
    buf_hot_pool: Buffer,
    buf_cold_pool: Buffer,
    buf_loom_refs: Buffer,
    buf_output: Buffer,
    buf_dictionary: Buffer,

    /// Capacities (in bytes) of each persistent buffer.
    cap_q_exact: u64,
    cap_q_latent: u64,
    cap_hot_pool: u64,
    cap_cold_pool: u64,
    cap_loom_refs: u64,
    cap_output: u64,
    cap_dictionary: u64,
}

/// Result of a single decode step
pub struct DecodeOutput {
    /// The output token: [q_heads * head_dim] f16 values
    pub data: Vec<F16>,
}

impl WeaverEngine {
    /// Default initial capacities sized for Llama-8B worst case.
    const INIT_Q_HEADS: u64 = 32;
    const INIT_KV_HEADS: u64 = 8;
    const INIT_HEAD_DIM: u64 = 128;
    const INIT_DICT_SIZE: u64 = 512;
    const INIT_MAX_HOT_TOKENS: u64 = 4096 * 16;
    const INIT_MAX_COLD_TOKENS: u64 = 32768 * 16;
    const INIT_MAX_TOTAL_BLOCKS: u64 = 4096 + 32768;

    /// Create a new WeaverEngine by loading the pre-compiled metallib.
    ///
    /// Pre-allocates persistent GPU buffers sized for typical workloads.
    ///
    /// # Arguments
    ///
    /// - `metallib_path`: Path to the compiled `weaver.metallib`
    ///
    /// # Errors
    ///
    /// Returns an error string if the Metal device, library, or pipeline creation fails.
    pub fn new(metallib_path: &str) -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found (Apple Silicon required)".to_string())?;

        let library = device.new_library_with_file(metallib_path)?;

        let function = library
            .get_function("weaver_decode", None)
            .map_err(|e| format!("Failed to load weaver_decode function: {}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)?;

        let queue = device.new_command_queue();

        // Pre-allocate persistent buffers
        let opts = MTLResourceOptions::StorageModeShared;
        let sc_size = std::mem::size_of::<SparseCode>() as u64;

        let cap_q_exact    = Self::INIT_Q_HEADS * Self::INIT_HEAD_DIM * 2;
        let cap_q_latent   = Self::INIT_Q_HEADS * Self::INIT_DICT_SIZE * 2;
        let cap_hot_pool   = Self::INIT_MAX_HOT_TOKENS * Self::INIT_KV_HEADS * Self::INIT_HEAD_DIM * 2;
        let cap_cold_pool  = Self::INIT_MAX_COLD_TOKENS * Self::INIT_KV_HEADS * sc_size;
        let cap_loom_refs  = Self::INIT_MAX_TOTAL_BLOCKS * 4;
        let cap_output     = Self::INIT_Q_HEADS * Self::INIT_HEAD_DIM * 2;
        let cap_dictionary = Self::INIT_KV_HEADS * Self::INIT_DICT_SIZE * Self::INIT_HEAD_DIM * 2;

        Ok(Self {
            buf_q_exact:    device.new_buffer(cap_q_exact,    opts),
            buf_q_latent:   device.new_buffer(cap_q_latent,   opts),
            buf_hot_pool:   device.new_buffer(cap_hot_pool,   opts),
            buf_cold_pool:  device.new_buffer(cap_cold_pool,  opts),
            buf_loom_refs:  device.new_buffer(cap_loom_refs,  opts),
            buf_output:     device.new_buffer(cap_output,     opts),
            buf_dictionary: device.new_buffer(cap_dictionary, opts),
            cap_q_exact, cap_q_latent, cap_hot_pool, cap_cold_pool,
            cap_loom_refs, cap_output, cap_dictionary,
            device,
            queue,
            pipeline,
            _library: library,
        })
    }

    /// Decode one token using the Weaver kernel.
    ///
    /// Data is copied into persistent GPU buffers via the shared-memory
    /// `contents()` pointer. Buffers are only reallocated if the input
    /// exceeds the pre-allocated capacity (rare path).
    pub fn decode(
        &mut self,
        q_exact: &[F16],
        q_latent: &[F16],
        hot_pool: &[F16],
        cold_pool: &[SparseCode],
        loom_refs: &[u32],
        dictionary: &[F16],
        params: &WeaverParams,
    ) -> Result<DecodeOutput, String> {
        let q_heads = params.q_heads as usize;
        let head_dim = params.head_dim as usize;
        let output_size = q_heads * head_dim;
        let opts = MTLResourceOptions::StorageModeShared;

        // Copy input data into persistent buffers, reallocating only on overflow.
        Self::fill_f16(&mut self.buf_q_exact,    &mut self.cap_q_exact,    q_exact,     &self.device, opts);
        Self::fill_f16(&mut self.buf_q_latent,   &mut self.cap_q_latent,   q_latent,    &self.device, opts);
        Self::fill_f16(&mut self.buf_hot_pool,   &mut self.cap_hot_pool,   hot_pool,    &self.device, opts);
        Self::fill_sparse(&mut self.buf_cold_pool, &mut self.cap_cold_pool, cold_pool,  &self.device, opts);
        Self::fill_u32(&mut self.buf_loom_refs,  &mut self.cap_loom_refs,  loom_refs,   &self.device, opts);
        Self::fill_f16(&mut self.buf_dictionary, &mut self.cap_dictionary, dictionary,  &self.device, opts);

        // Ensure output buffer is large enough
        let output_bytes = (output_size * std::mem::size_of::<F16>()) as u64;
        if output_bytes > self.cap_output {
            self.buf_output = self.device.new_buffer(output_bytes, opts);
            self.cap_output = output_bytes;
        }

        // Encode and dispatch
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.buf_q_exact),    0);
        encoder.set_buffer(1, Some(&self.buf_q_latent),   0);
        encoder.set_buffer(2, Some(&self.buf_hot_pool),   0);
        encoder.set_buffer(3, Some(&self.buf_cold_pool),  0);
        encoder.set_buffer(4, Some(&self.buf_loom_refs),  0);
        encoder.set_buffer(5, Some(&self.buf_output),     0);

        let params_ptr = params as *const WeaverParams as *const c_void;
        let params_size = std::mem::size_of::<WeaverParams>() as u64;
        encoder.set_bytes(6, params_size, params_ptr);
        encoder.set_buffer(7, Some(&self.buf_dictionary), 0);

        let grid_size = MTLSize::new(1, q_heads as u64, 1);
        let threadgroup_size = MTLSize::new(32, 1, 1);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back from persistent output buffer
        let output_ptr = self.buf_output.contents() as *const F16;
        let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, output_size) };

        Ok(DecodeOutput { data: output_slice.to_vec() })
    }

    // ─── Persistent buffer fill helpers ──────────────────────────────────

    fn fill_f16(buf: &mut Buffer, cap: &mut u64, data: &[F16], dev: &Device, opts: MTLResourceOptions) {
        let needed = (data.len() * 2) as u64;
        if needed == 0 { return; }
        if needed > *cap || *cap == 0 { *buf = dev.new_buffer(needed, opts); *cap = needed; }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents() as *mut u8, needed as usize); }
    }

    fn fill_u32(buf: &mut Buffer, cap: &mut u64, data: &[u32], dev: &Device, opts: MTLResourceOptions) {
        let needed = (data.len() * 4) as u64;
        if needed == 0 { return; }
        if needed > *cap || *cap == 0 { *buf = dev.new_buffer(needed, opts); *cap = needed; }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents() as *mut u8, needed as usize); }
    }

    fn fill_sparse(buf: &mut Buffer, cap: &mut u64, data: &[SparseCode], dev: &Device, opts: MTLResourceOptions) {
        let needed = (data.len() * std::mem::size_of::<SparseCode>()) as u64;
        if needed == 0 { return; }
        if needed > *cap || *cap == 0 { *buf = dev.new_buffer(needed, opts); *cap = needed; }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents() as *mut u8, needed as usize); }
    }

    /// Get the Metal device name
    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Get the pipeline's max total threads per threadgroup
    pub fn max_threads_per_threadgroup(&self) -> u64 {
        self.pipeline.max_total_threads_per_threadgroup()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16 as F16;

    /// Verify WeaverEngine can be created on Apple Silicon
    #[test]
    fn weaver_engine_creates_pipeline() {
        let metallib_path = option_env!("WEAVER_METALLIB");
        if metallib_path.is_none() {
            eprintln!("WEAVER_METALLIB not set — skipping pipeline test (normal on CI)");
            return;
        }

        let engine = WeaverEngine::new(metallib_path.unwrap());
        assert!(engine.is_ok(), "Pipeline creation failed: {:?}", engine.err());

        let engine = engine.unwrap();
        println!("Device: {}", engine.device_name());
        println!("Max threads/threadgroup: {}", engine.max_threads_per_threadgroup());
        assert!(engine.max_threads_per_threadgroup() >= 32);
    }

    /// Verify a single decode step produces non-zero output
    #[test]
    fn weaver_decode_produces_output() {
        let metallib_path = option_env!("WEAVER_METALLIB");
        if metallib_path.is_none() {
            eprintln!("WEAVER_METALLIB not set — skipping decode test");
            return;
        }

        let mut engine = WeaverEngine::new(metallib_path.unwrap()).unwrap();

        let q_heads: u32 = 32;
        let kv_heads: u32 = 8;
        let head_dim: u32 = 128;
        let dict_size: u32 = 512;
        let hot_count: u32 = 2;
        let cold_count: u32 = 1;
        let block_size: u32 = 16;

        let one = F16::from_f32(1.0);
        let q_exact = vec![one; (q_heads * head_dim) as usize];
        let q_latent = vec![one; (q_heads * dict_size) as usize];
        let hot_pool = vec![one; (hot_count * block_size * kv_heads * head_dim) as usize];
        let cold_codes = vec![SparseCode::zero(); (cold_count * block_size * kv_heads) as usize];
        let loom_refs: Vec<u32> = (0..(hot_count + cold_count)).collect();
        let dictionary = vec![one; (kv_heads * dict_size * head_dim) as usize];

        let params = WeaverParams {
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            gqa_group: q_heads / kv_heads,
            hot_count,
            cold_count,
            dict_size,
            sparsity_k: 4,
            _pad: [0; 3],
        };

        let result = engine.decode(
            &q_exact, &q_latent, &hot_pool, &cold_codes,
            &loom_refs, &dictionary, &params,
        );

        assert!(result.is_ok(), "Decode failed: {:?}", result.err());

        let output = result.unwrap();
        assert_eq!(output.data.len(), (q_heads * head_dim) as usize);

        let has_nonzero = output.data.iter().any(|v| f32::from(*v) != 0.0);
        assert!(has_nonzero, "Output should contain non-zero values");

        println!("Decode output sample (first 8 values):");
        for (i, v) in output.data.iter().take(8).enumerate() {
            println!("  [{}] = {:.4}", i, f32::from(*v));
        }
    }
}
