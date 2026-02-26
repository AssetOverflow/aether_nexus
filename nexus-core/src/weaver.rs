//! WeaverEngine – Rust-side Metal compute pipeline for the Weaver decode kernel.
//!
//! Orchestrates GPU dispatch of the Weaver kernel:
//! 1. Load the pre-compiled metallib (from build.rs)
//! 2. Create compute pipeline state
//! 3. Allocate GPU buffers for kernel inputs/outputs
//! 4. Encode and dispatch the kernel
//! 5. Read back the output token

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
pub struct WeaverEngine {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    _library: Library,
}

/// Result of a single decode step
pub struct DecodeOutput {
    /// The output token: [q_heads * head_dim] f16 values
    pub data: Vec<F16>,
}

impl WeaverEngine {
    /// Create a new WeaverEngine by loading the pre-compiled metallib.
    ///
    /// # Arguments
    ///
    /// - `metallib_path`: Path to the compiled `weaver.metallib`
    ///
    /// # Errors
    ///
    /// Returns an error string if the Metal device, library, or pipeline creation fails.
    pub fn new(metallib_path: &str) -> Result<Self, String> {
        // 1. Get the system default Metal device
        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found (Apple Silicon required)".to_string())?;

        // 2. Load the pre-compiled metallib
        let library = device.new_library_with_file(metallib_path)?;

        // 3. Get the kernel function
        let function = library
            .get_function("weaver_decode", None)
            .map_err(|e| format!("Failed to load weaver_decode function: {}", e))?;

        // 4. Create the compute pipeline state
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)?;

        // 5. Create command queue
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            queue,
            pipeline,
            _library: library,
        })
    }

    /// Decode one token using the Weaver kernel.
    ///
    /// # Arguments
    ///
    /// All data must be pre-allocated and correctly sized:
    /// - `q_exact`: Query vectors for hot path [q_heads * head_dim] f16
    /// - `q_latent`: Query projected into dictionary space [q_heads * dict_size] f16
    /// - `hot_pool`: Hot KV blocks [n_total_hot_tokens * kv_heads * head_dim] f16
    /// - `cold_pool`: Cold SparseCode blocks [n_total_cold_tokens * kv_heads] SparseCode
    /// - `loom_refs`: Block reference indices [hot_count + cold_count] u32
    /// - `dictionary`: Learned dictionary [kv_heads * dict_size * head_dim] f16
    /// - `params`: Kernel dispatch parameters
    ///
    /// # Returns
    ///
    /// The decoded output token as f16 values.
    pub fn decode(
        &self,
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

        // Create GPU buffers
        let opts = MTLResourceOptions::StorageModeShared;

        let buf_q_exact = self.create_buffer_f16(q_exact, opts);
        let buf_q_latent = self.create_buffer_f16(q_latent, opts);
        let buf_hot_pool = self.create_buffer_f16(hot_pool, opts);
        let buf_cold_pool = self.create_buffer_sparse(cold_pool, opts);
        let buf_loom_refs = self.create_buffer_u32(loom_refs, opts);
        let buf_output = self.device.new_buffer(
            (output_size * std::mem::size_of::<F16>()) as u64,
            opts,
        );
        let buf_dictionary = self.create_buffer_f16(dictionary, opts);

        // Create command buffer and encoder
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        // Set pipeline
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Bind buffers (must match kernel signature)
        encoder.set_buffer(0, Some(&buf_q_exact), 0);
        encoder.set_buffer(1, Some(&buf_q_latent), 0);
        encoder.set_buffer(2, Some(&buf_hot_pool), 0);
        encoder.set_buffer(3, Some(&buf_cold_pool), 0);
        encoder.set_buffer(4, Some(&buf_loom_refs), 0);
        encoder.set_buffer(5, Some(&buf_output), 0);

        // Bind params as constant buffer (buffer 6)
        let params_ptr = params as *const WeaverParams as *const c_void;
        let params_size = std::mem::size_of::<WeaverParams>() as u64;
        encoder.set_bytes(6, params_size, params_ptr);

        // Bind dictionary (buffer 7)
        encoder.set_buffer(7, Some(&buf_dictionary), 0);

        // Dispatch: grid = (1, q_heads, 1), threadgroup = (32, 1, 1)
        let grid_size = MTLSize::new(1, q_heads as u64, 1);
        let threadgroup_size = MTLSize::new(32, 1, 1);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);

        encoder.end_encoding();

        // Submit and wait
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back output
        let output_ptr = buf_output.contents() as *const F16;
        let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, output_size) };
        let output_data = output_slice.to_vec();

        Ok(DecodeOutput { data: output_data })
    }

    // ─── Buffer creation helpers ─────────────────────────────────────────

    fn create_buffer_f16(&self, data: &[F16], opts: MTLResourceOptions) -> Buffer {
        let ptr = data.as_ptr() as *const c_void;
        let len = (data.len() * std::mem::size_of::<F16>()) as u64;
        self.device.new_buffer_with_data(ptr, len, opts)
    }

    fn create_buffer_u32(&self, data: &[u32], opts: MTLResourceOptions) -> Buffer {
        let ptr = data.as_ptr() as *const c_void;
        let len = (data.len() * std::mem::size_of::<u32>()) as u64;
        self.device.new_buffer_with_data(ptr, len, opts)
    }

    fn create_buffer_sparse(&self, data: &[SparseCode], opts: MTLResourceOptions) -> Buffer {
        let ptr = data.as_ptr() as *const c_void;
        let len = (data.len() * std::mem::size_of::<SparseCode>()) as u64;
        self.device.new_buffer_with_data(ptr, len, opts)
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
        // The metallib should be at the build output path
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

        let engine = WeaverEngine::new(metallib_path.unwrap()).unwrap();

        let q_heads: u32 = 32;
        let kv_heads: u32 = 8;
        let head_dim: u32 = 128;
        let dict_size: u32 = 512;
        let hot_count: u32 = 2;
        let cold_count: u32 = 1;
        let block_size: u32 = 16;

        // Create synthetic test data
        let one = F16::from_f32(1.0);
        let q_exact = vec![one; (q_heads * head_dim) as usize];
        let q_latent = vec![one; (q_heads * dict_size) as usize];

        // Hot pool: 2 blocks × block_size × kv_heads × head_dim
        let hot_pool = vec![one; (hot_count * block_size * kv_heads * head_dim) as usize];

        // Cold pool: 1 block × block_size × kv_heads
        let cold_codes = vec![SparseCode::zero(); (cold_count * block_size * kv_heads) as usize];

        // Loom refs: indices for hot + cold blocks
        let loom_refs: Vec<u32> = (0..(hot_count + cold_count)).collect();

        // Dictionary: kv_heads × dict_size × head_dim
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
            &q_exact,
            &q_latent,
            &hot_pool,
            &cold_codes,
            &loom_refs,
            &dictionary,
            &params,
        );

        assert!(result.is_ok(), "Decode failed: {:?}", result.err());

        let output = result.unwrap();
        assert_eq!(output.data.len(), (q_heads * head_dim) as usize);

        // With uniform inputs, output should be non-zero
        let has_nonzero = output.data.iter().any(|v| f32::from(*v) != 0.0);
        assert!(has_nonzero, "Output should contain non-zero values");

        println!("Decode output sample (first 8 values):");
        for (i, v) in output.data.iter().take(8).enumerate() {
            println!("  [{}] = {:.4}", i, f32::from(*v));
        }
    }
}
