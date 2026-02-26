//! GPU Ops Engine – Rust-side dispatch for transformer Metal kernels.
//!
//! Wraps the ops.metallib kernels for use in the forward pass.
//! Supports batch mode: `begin_batch()` / `end_batch()` to encode
//! all operations into a single command buffer for maximum throughput.

use half::f16 as F16;
use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions,
    MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

/// GPU Ops Engine – manages the Metal compute pipeline for transformer ops.
pub struct OpsEngine {
    device: Device,
    queue: CommandQueue,
    kernels: HashMap<&'static str, ComputePipelineState>,
    _library: Library,
    /// When Some, all ops encode into this shared command buffer (batch mode).
    batch_cmd_buf: Mutex<Option<CommandBuffer>>,
}

impl OpsEngine {
    /// Create a new OpsEngine by loading the ops.metallib.
    pub fn new(metallib_path: &str) -> Result<Self, String> {
        let device = Device::system_default().ok_or_else(|| "No Metal device found".to_string())?;
        let library = device.new_library_with_file(metallib_path)?;
        let queue = device.new_command_queue();

        let kernel_names = [
            "copy_buffer",
            "embed_lookup",
            "rms_norm",
            "rope",
            "matmul_f16",
            "vecmat_f16",
            "vecmat_scaled",
            "silu_gate",
            "add_residual",
            "scale_f16",
            "matmul_scaled",
            "causal_attention",
            "multihead_attention",
        ];

        let mut kernels = HashMap::new();
        for name in &kernel_names {
            let func = library
                .get_function(name, None)
                .map_err(|e| format!("Failed to load kernel '{}': {}", name, e))?;
            let pipeline = device.new_compute_pipeline_state_with_function(&func)?;
            kernels.insert(*name, pipeline);
        }

        Ok(Self {
            device,
            queue,
            kernels,
            _library: library,
            batch_cmd_buf: Mutex::new(None),
        })
    }

    // ─── Batch mode ─────────────────────────────────────────────────────

    /// Start batch mode: all subsequent ops encode into ONE command buffer.
    pub fn begin_batch(&self) {
        let cmd_buf = self.queue.new_command_buffer();
        *self.batch_cmd_buf.lock().unwrap() = Some(cmd_buf.to_owned());
    }

    /// End batch mode: commit the shared command buffer and wait for GPU.
    pub fn end_batch(&self) {
        if let Some(cmd_buf) = self.batch_cmd_buf.lock().unwrap().take() {
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }
    }

    /// Internal: encode a compute pass. In batch mode, uses the shared
    /// command buffer. Otherwise creates a standalone one.
    fn encode_compute<F: FnOnce(&metal::ComputeCommandEncoderRef)>(&self, f: F) {
        let batch = self.batch_cmd_buf.lock().unwrap();
        if let Some(ref cmd_buf) = *batch {
            let encoder = cmd_buf.new_compute_command_encoder();
            f(encoder);
            encoder.end_encoding();
            // Don't commit — batch mode, will commit in end_batch()
        } else {
            drop(batch); // release borrow
            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            f(encoder);
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }
    }

    // ─── Buffer helpers ─────────────────────────────────────────────────

    /// Create a GPU buffer from f16 data.
    pub fn buffer_f16(&self, data: &[F16]) -> Buffer {
        let ptr = data.as_ptr() as *const c_void;
        let len = (data.len() * 2) as u64;
        self.device
            .new_buffer_with_data(ptr, len, MTLResourceOptions::StorageModeShared)
    }

    /// Create an empty GPU buffer of given byte size.
    pub fn buffer_empty(&self, bytes: u64) -> Buffer {
        self.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    /// Create a GPU buffer from u32 data.
    pub fn buffer_u32(&self, data: &[u32]) -> Buffer {
        let ptr = data.as_ptr() as *const c_void;
        let len = (data.len() * 4) as u64;
        self.device
            .new_buffer_with_data(ptr, len, MTLResourceOptions::StorageModeShared)
    }

    /// Read f16 data back from a GPU buffer.
    pub fn read_f16(&self, buffer: &Buffer, count: usize) -> Vec<F16> {
        let ptr = buffer.contents() as *const F16;
        unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
    }

    /// Read f32 data back from a GPU buffer.
    pub fn read_f32(&self, buffer: &Buffer, count: usize) -> Vec<f32> {
        let ptr = buffer.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
    }

    // ─── Kernel dispatch methods ─────────────────────────────────────────

    /// GPU buffer copy: dst[dst_off..dst_off+count] = src[src_off..src_off+count]
    /// Offsets are in f16 ELEMENTS, not bytes.
    pub fn copy_buffer(&self, src: &Buffer, dst: &Buffer, count: u32, src_off: u32, dst_off: u32) {
        self.encode_compute(|encoder| {
            encoder.set_compute_pipeline_state(&self.kernels["copy_buffer"]);
            encoder.set_buffer(0, Some(src), 0);
            encoder.set_buffer(1, Some(dst), 0);
            encoder.set_bytes(2, 4, &count as *const u32 as *const c_void);
            encoder.set_bytes(3, 4, &src_off as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &dst_off as *const u32 as *const c_void);
            let tg_size = 256u64.min(count as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new((count as u64 + tg_size - 1) / tg_size, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });
    }

    /// Embedding lookup: output[i] = embed_table[token_ids[i]]
    pub fn embed_lookup(
        &self,
        token_ids: &Buffer,
        embed_table: &Buffer,
        output: &Buffer,
        hidden_size: u32,
        seq_len: u32,
    ) {
        let hs = hidden_size;
        let sl = seq_len;
        self.encode_compute(|encoder| {
            encoder.set_compute_pipeline_state(&self.kernels["embed_lookup"]);
            encoder.set_buffer(0, Some(token_ids), 0);
            encoder.set_buffer(1, Some(embed_table), 0);
            encoder.set_buffer(2, Some(output), 0);
            encoder.set_bytes(3, 4, &hs as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &sl as *const u32 as *const c_void);
            let tg = MTLSize::new(hs.min(256) as u64, 1, 1);
            encoder.dispatch_thread_groups(
                MTLSize::new((hs as u64 + tg.width - 1) / tg.width, sl as u64, 1),
                tg,
            );
        });
    }

    /// RMS norm: output = input * rsqrt(mean(input²) + eps) * weight
    pub fn rms_norm(
        &self,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        hidden_size: u32,
        num_tokens: u32,
        eps: f32,
    ) {
        self.encode_compute(|encoder| {
            encoder.set_compute_pipeline_state(&self.kernels["rms_norm"]);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(weight), 0);
            encoder.set_buffer(2, Some(output), 0);
            encoder.set_bytes(3, 4, &hidden_size as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &eps as *const f32 as *const c_void);
            let tgs = 256u64.min(hidden_size as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new(num_tokens as u64, 1, 1),
                MTLSize::new(tgs, 1, 1),
            );
        });
    }

    /// RoPE: apply rotary embeddings to Q and K
    pub fn rope(
        &self,
        q: &Buffer,
        k: &Buffer,
        seq_len: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        position: u32,
        theta: f32,
    ) {
        self.encode_compute(|encoder| {
            encoder.set_compute_pipeline_state(&self.kernels["rope"]);
            encoder.set_buffer(0, Some(q), 0);
            encoder.set_buffer(1, Some(k), 0);
            encoder.set_bytes(2, 4, &seq_len as *const u32 as *const c_void);
            encoder.set_bytes(3, 4, &q_heads as *const u32 as *const c_void);
            encoder.set_bytes(4, 4, &kv_heads as *const u32 as *const c_void);
            encoder.set_bytes(5, 4, &head_dim as *const u32 as *const c_void);
            encoder.set_bytes(6, 4, &position as *const u32 as *const c_void);
            encoder.set_bytes(7, 4, &theta as *const f32 as *const c_void);
            let half_dim = head_dim / 2;
            let max_heads = q_heads.max(kv_heads);
            encoder.dispatch_thread_groups(
                MTLSize::new(((half_dim + 31) / 32) as u64, max_heads as u64, 1),
                MTLSize::new(32.min(half_dim as u64), 1, 1),
            );
        });
    }

    /// MatMul: C = A × B^T, where A:[M,K], B:[N,K], C:[M,N], all f16.
    /// Auto-selects kernel: vecmat_f16 for M=1 (decode), tiled matmul_f16 for M>1.
    pub fn matmul(&self, a: &Buffer, b: &Buffer, output: &Buffer, m: u32, n: u32, k: u32) {
        if m == 1 {
            // M=1: SIMD vec-mat kernel. 32 threads/SIMD group, 8 columns per threadgroup.
            self.encode_compute(|encoder| {
                encoder.set_compute_pipeline_state(&self.kernels["vecmat_f16"]);
                encoder.set_buffer(0, Some(a), 0);
                encoder.set_buffer(1, Some(b), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &n as *const u32 as *const c_void);
                encoder.set_bytes(4, 4, &k as *const u32 as *const c_void);
                let tg = MTLSize::new(256, 1, 1);
                encoder.dispatch_thread_groups(MTLSize::new((n as u64 + 7) / 8, 1, 1), tg);
            });
        } else {
            // M>1: use tiled 16×16 shared-memory kernel
            self.encode_compute(|encoder| {
                encoder.set_compute_pipeline_state(&self.kernels["matmul_f16"]);
                encoder.set_buffer(0, Some(a), 0);
                encoder.set_buffer(1, Some(b), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &m as *const u32 as *const c_void);
                encoder.set_bytes(4, 4, &n as *const u32 as *const c_void);
                encoder.set_bytes(5, 4, &k as *const u32 as *const c_void);
                let tg = MTLSize::new(16, 16, 1);
                encoder.dispatch_thread_groups(
                    MTLSize::new((n as u64 + 15) / 16, (m as u64 + 15) / 16, 1),
                    tg,
                );
            });
        }
    }

    /// SiLU gate: output = SiLU(gate) * up
    pub fn silu_gate(&self, gate: &Buffer, up: &Buffer, output: &Buffer, size: u32) {
        self.encode_compute(|encoder| {
            encoder.set_compute_pipeline_state(&self.kernels["silu_gate"]);
            encoder.set_buffer(0, Some(gate), 0);
            encoder.set_buffer(1, Some(up), 0);
            encoder.set_buffer(2, Some(output), 0);
            encoder.set_bytes(3, 4, &size as *const u32 as *const c_void);
            let tg_size = 256u64.min(size as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new((size as u64 + tg_size - 1) / tg_size, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });
    }

    /// Add residual in place: x += residual
    pub fn add_residual(&self, x: &Buffer, residual: &Buffer, size: u32) {
        self.encode_compute(|encoder| {
            encoder.set_compute_pipeline_state(&self.kernels["add_residual"]);
            encoder.set_buffer(0, Some(x), 0);
            encoder.set_buffer(1, Some(residual), 0);
            encoder.set_bytes(2, 4, &size as *const u32 as *const c_void);
            let tg_size = 256u64.min(size as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new((size as u64 + tg_size - 1) / tg_size, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });
    }

    /// Scale f16 buffer in-place: x[i] *= scale
    ///
    /// Eliminates CPU↔GPU round-trips for scalar multiplication.
    /// Granite uses 3 scaling ops per layer (embedding, attention, residual)
    /// which previously required ~121 GPU syncs per token. With this kernel,
    /// all scaling stays within a single GPU command buffer batch.
    pub fn scale_f16(&self, x: &Buffer, count: u32, scale: f32) {
        self.encode_compute(|encoder| {
            encoder.set_compute_pipeline_state(&self.kernels["scale_f16"]);
            encoder.set_buffer(0, Some(x), 0);
            encoder.set_bytes(1, 4, &count as *const u32 as *const c_void);
            encoder.set_bytes(2, 4, &scale as *const f32 as *const c_void);
            let tg_size = 256u64.min(count as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new((count as u64 + tg_size - 1) / tg_size, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });
    }

    /// Scaled matmul for logits: C = A × B^T * scale, output in f32.
    /// Auto-selects kernel: vecmat_scaled for M=1, tiled matmul_scaled for M>1.
    pub fn matmul_scaled(
        &self,
        a: &Buffer,
        b: &Buffer,
        output: &Buffer,
        m: u32,
        n: u32,
        k: u32,
        scale: f32,
    ) {
        if m == 1 {
            // M=1: SIMD vec-mat kernel with scale, 8 columns per threadgroup.
            self.encode_compute(|encoder| {
                encoder.set_compute_pipeline_state(&self.kernels["vecmat_scaled"]);
                encoder.set_buffer(0, Some(a), 0);
                encoder.set_buffer(1, Some(b), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &n as *const u32 as *const c_void);
                encoder.set_bytes(4, 4, &k as *const u32 as *const c_void);
                encoder.set_bytes(5, 4, &scale as *const f32 as *const c_void);
                let tg = MTLSize::new(256, 1, 1);
                encoder.dispatch_thread_groups(MTLSize::new((n as u64 + 7) / 8, 1, 1), tg);
            });
        } else {
            self.encode_compute(|encoder| {
                encoder.set_compute_pipeline_state(&self.kernels["matmul_scaled"]);
                encoder.set_buffer(0, Some(a), 0);
                encoder.set_buffer(1, Some(b), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &m as *const u32 as *const c_void);
                encoder.set_bytes(4, 4, &n as *const u32 as *const c_void);
                encoder.set_bytes(5, 4, &k as *const u32 as *const c_void);
                encoder.set_bytes(6, 4, &scale as *const f32 as *const c_void);
                let tg = MTLSize::new(16, 16, 1);
                encoder.dispatch_thread_groups(
                    MTLSize::new((n as u64 + 15) / 16, (m as u64 + 15) / 16, 1),
                    tg,
                );
            });
        }
    }

    /// Causal self-attention for a single head (legacy, kept for compatibility).
    pub fn causal_attention(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        output: &Buffer,
        kv_len: u32,
        head_dim: u32,
    ) {
        self.encode_compute(|encoder| {
            encoder.set_compute_pipeline_state(&self.kernels["causal_attention"]);
            encoder.set_buffer(0, Some(q), 0);
            encoder.set_buffer(1, Some(k), 0);
            encoder.set_buffer(2, Some(v), 0);
            encoder.set_buffer(3, Some(output), 0);
            encoder.set_bytes(4, 4, &kv_len as *const u32 as *const c_void);
            encoder.set_bytes(5, 4, &head_dim as *const u32 as *const c_void);
            let tg_size = 64u64.min(head_dim as u64);
            encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_size, 1, 1));
        });
    }

    /// Multi-head GQA attention: all query heads in ONE dispatch.
    /// Q: [q_heads * head_dim], K/V_cache: [seq_len * kv_heads * head_dim]
    pub fn multihead_attention(
        &self,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        output: &Buffer,
        kv_len: u32,
        head_dim: u32,
        q_heads: u32,
        kv_heads: u32,
    ) {
        self.encode_compute(|encoder| {
            encoder.set_compute_pipeline_state(&self.kernels["multihead_attention"]);
            encoder.set_buffer(0, Some(q), 0);
            encoder.set_buffer(1, Some(k_cache), 0);
            encoder.set_buffer(2, Some(v_cache), 0);
            encoder.set_buffer(3, Some(output), 0);
            encoder.set_bytes(4, 4, &kv_len as *const u32 as *const c_void);
            encoder.set_bytes(5, 4, &head_dim as *const u32 as *const c_void);
            encoder.set_bytes(6, 4, &q_heads as *const u32 as *const c_void);
            encoder.set_bytes(7, 4, &kv_heads as *const u32 as *const c_void);
            let tg_size = 64u64.min(head_dim as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new(q_heads as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
        });
    }
}
