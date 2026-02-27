//! Inference engine for Granite / Llama-style transformers.
//!
//! Implements the full autoregressive decode loop:
//!   embed → (RMSNorm → Attn → Residual → RMSNorm → FFN → Residual) × N → Logits → Sample

use crate::ops::OpsEngine;
use crate::tokenizer::Tokenizer;
use crate::weight_loader::ModelWeights;
// half::f16 as F16 removed — scaling is now GPU-side via scale_f16 kernel
use metal::Buffer;

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Granite-specific: multiply embeddings by this factor
    pub embedding_multiplier: f32,
    /// Granite-specific: multiply attention by this factor  
    pub attention_multiplier: f32,
    /// Granite-specific: multiply residual connections by this factor
    pub residual_multiplier: f32,
    /// Granite-specific: scale logits by this factor
    pub logits_scaling: f32,
    /// Sampling temperature
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// End-of-sequence token ID (model-specific)
    pub eos_token_id: u32,
    /// Turn-end token ID used for chat interruptions (<|im_end|>)
    pub im_end_token_id: u32,
    /// Human-readable model name for logging
    pub model_name: String,
}

impl InferenceConfig {
    /// Config for Granite 3.0 2B Instruct
    pub fn granite_2b() -> Self {
        Self {
            num_layers: 40,
            hidden_size: 2048,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 64,
            intermediate_size: 8192,
            vocab_size: 49155,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            embedding_multiplier: 12.0,
            attention_multiplier: 0.015625, // 1/64
            residual_multiplier: 0.22,
            logits_scaling: 8.0,
            temperature: 0.7,
            max_tokens: 128,
            eos_token_id: 0,
            im_end_token_id: 0,
            model_name: "Granite 3.0 2B Instruct".into(),
        }
    }

    /// Config for Qwen 2.5 0.5B Instruct (System 1 – fast reflex)
    pub fn qwen_0_5b() -> Self {
        Self {
            num_layers: 24,
            hidden_size: 896,
            q_heads: 14,
            kv_heads: 2,
            head_dim: 64, // 896 / 14 = 64
            intermediate_size: 4864,
            vocab_size: 151936,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            embedding_multiplier: 1.0,
            attention_multiplier: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            temperature: 0.7,
            max_tokens: 256,
            eos_token_id: 151645, // <|im_end|>
            im_end_token_id: 151645,
            model_name: "Qwen 2.5 0.5B Instruct (System 1)".into(),
        }
    }

    /// Config for DeepSeek-R1-Distill-Qwen-1.5B (System 2 – deep reasoning)
    pub fn deepseek_r1_1_5b() -> Self {
        Self {
            num_layers: 28,
            hidden_size: 1536,
            q_heads: 12,
            kv_heads: 2,
            head_dim: 128, // 1536 / 12 = 128
            intermediate_size: 8960,
            vocab_size: 151936,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            embedding_multiplier: 1.0,
            attention_multiplier: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            temperature: 0.6,        // Slightly lower for more focused reasoning
            max_tokens: 512,         // Longer context for reasoning chains
            eos_token_id: 151643,    // DeepSeek uses bos_token_id as eos
            im_end_token_id: 151645, // <|im_end|>
            model_name: "DeepSeek-R1-Distill-Qwen-1.5B (System 2)".into(),
        }
    }

    /// Auto-detect model type from a config.json file in the model directory.
    pub fn detect_from_dir(model_dir: &str) -> Result<Self, String> {
        let config_path = std::path::Path::new(model_dir).join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read {}: {}", config_path.display(), e))?;

        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let arch = config["architectures"]
            .as_array()
            .and_then(|arr| arr.get(0))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        let model_type = config["model_type"].as_str().unwrap_or("unknown");
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(0) as usize;
        let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(0) as usize;

        if (arch == "GraniteForCausalLM" || arch == "LlamaForCausalLM" || model_type == "granite")
            && hidden_size == 2048
        {
            println!("[DETECT] Matched: Granite 3.0 2B Instruct");
            Ok(Self::granite_2b())
        } else if arch == "Qwen2ForCausalLM" || model_type == "qwen2" {
            if hidden_size == 896 {
                println!("[DETECT] Matched: Qwen 2.5 0.5B Instruct");
                Ok(Self::qwen_0_5b())
            } else if hidden_size == 1536 {
                println!("[DETECT] Matched: DeepSeek-R1-Distill-Qwen-1.5B");
                Ok(Self::deepseek_r1_1_5b())
            } else {
                Err(format!(
                    "Unknown Qwen2 model size: hidden_size={}, num_layers={}.",
                    hidden_size, num_layers
                ))
            }
        } else {
            Err(format!(
                "Unknown model architecture: arch={}, type={}, hidden_size={}, num_layers={}. Add a config preset.",
                arch, model_type, hidden_size, num_layers
            ))
        }
    }
}

struct KvCache {
    /// [max_seq, kv_heads * head_dim] f16
    k_cache: Buffer,
    /// [max_seq, kv_heads * head_dim] f16  
    v_cache: Buffer,
}

/// The inference engine holds GPU weight buffers and KV caches.
pub struct InferenceEngine {
    ops: OpsEngine,
    pub config: InferenceConfig,
    // Weight buffers (uploaded to GPU once)
    buf_embed: Buffer,
    buf_lm_head: Buffer,
    buf_final_norm: Buffer,
    layer_bufs: Vec<LayerBuffers>,
    // KV caches per layer
    kv_caches: Vec<KvCache>,
    // Workspace buffers to avoid reallocation
    ws_token: Buffer,
    ws_hidden: Buffer,
    ws_norm: Buffer,
    ws_q: Buffer,
    ws_k: Buffer,
    ws_v: Buffer,
    ws_attn_out: Buffer,
    ws_proj: Buffer,
    ws_gate: Buffer,
    ws_up: Buffer,
    ws_ffn: Buffer,
    ws_down: Buffer,
    ws_residual: Buffer,
    ws_logits: Buffer,

    /// Current sequence position
    position: u32,
}

/// GPU buffers for one transformer layer's weights.
struct LayerBuffers {
    q_proj: Buffer,
    k_proj: Buffer,
    v_proj: Buffer,
    o_proj: Buffer,
    gate_proj: Buffer,
    up_proj: Buffer,
    down_proj: Buffer,
    input_layernorm: Buffer,
    post_attention_layernorm: Buffer,
    /// Optional Q/K/V biases (Qwen-style)
    q_bias: Option<Buffer>,
    k_bias: Option<Buffer>,
    v_bias: Option<Buffer>,
}

impl InferenceEngine {
    /// Maximum sequence length for KV cache allocation.
    /// Beyond this, `forward_batch` returns an error.
    pub const MAX_SEQ_LEN: usize = 4096;
    /// Maximum batch size for multi-token / speculative forward passes.
    pub const MAX_BATCH: usize = 16;

    /// Create an inference engine, uploading all weights to GPU.
    pub fn new(ops: OpsEngine, weights: &ModelWeights, config: InferenceConfig) -> Self {
        crate::nexus_info!("Uploading weights to GPU...");

        let buf_embed = ops.buffer_f16(&weights.embed_tokens);
        let buf_lm_head = ops.buffer_f16(&weights.lm_head);
        let buf_final_norm = ops.buffer_f16(&weights.final_norm);

        let mut layer_bufs = Vec::with_capacity(config.num_layers);
        for (i, layer) in weights.layers.iter().enumerate() {
            if i % 10 == 0 {
                crate::nexus_debug!(
                    "  Uploading layer {}/{}...",
                    i, config.num_layers
                );
            }
            layer_bufs.push(LayerBuffers {
                q_proj: ops.buffer_f16(&layer.q_proj),
                k_proj: ops.buffer_f16(&layer.k_proj),
                v_proj: ops.buffer_f16(&layer.v_proj),
                o_proj: ops.buffer_f16(&layer.o_proj),
                gate_proj: ops.buffer_f16(&layer.gate_proj),
                up_proj: ops.buffer_f16(&layer.up_proj),
                down_proj: ops.buffer_f16(&layer.down_proj),
                input_layernorm: ops.buffer_f16(&layer.input_layernorm),
                post_attention_layernorm: ops.buffer_f16(&layer.post_attention_layernorm),
                q_bias: layer.q_bias.as_ref().map(|b| ops.buffer_f16(b)),
                k_bias: layer.k_bias.as_ref().map(|b| ops.buffer_f16(b)),
                v_bias: layer.v_bias.as_ref().map(|b| ops.buffer_f16(b)),
            });
        }

        // Allocate KV caches
        let max_seq = Self::MAX_SEQ_LEN as u64;
        let kv_dim = (config.kv_heads * config.head_dim) as u64;
        let cache_size = max_seq * kv_dim * 2; // f16 = 2 bytes

        let mut kv_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            kv_caches.push(KvCache {
                k_cache: ops.buffer_empty(cache_size),
                v_cache: ops.buffer_empty(cache_size),
            });
        }

        crate::nexus_info!(
            "All weights uploaded. KV cache allocated for {} tokens.",
            max_seq
        );

        // Workspace sizes
        let h = config.hidden_size as u64;
        let q_h = config.q_heads as u64;
        let _kv_h = config.kv_heads as u64;
        let hd = config.head_dim as u64;
        let inter = config.intermediate_size as u64;
        let vocab = config.vocab_size as u64;
        
        // Scale workspaces for batch size
        let mb = Self::MAX_BATCH as u64;

        let ws_token = ops.buffer_empty(4 * mb); // MAX_BATCH u32 tokens
        let ws_hidden = ops.buffer_empty(h * 2 * mb);
        let ws_norm = ops.buffer_empty(h * 2 * mb);
        let ws_q = ops.buffer_empty(q_h * hd * 2 * mb);
        let ws_k = ops.buffer_empty(kv_dim * 2 * mb);
        let ws_v = ops.buffer_empty(kv_dim * 2 * mb);
        let ws_attn_out = ops.buffer_empty(q_h * hd * 2 * mb);
        let ws_proj = ops.buffer_empty(h * 2 * mb);
        let ws_gate = ops.buffer_empty(inter * 2 * mb);
        let ws_up = ops.buffer_empty(inter * 2 * mb);
        let ws_ffn = ops.buffer_empty(inter * 2 * mb);
        let ws_down = ops.buffer_empty(h * 2 * mb);
        let ws_residual = ops.buffer_empty(h * 2 * mb);
        
        let ws_logits = ops.buffer_empty(vocab * 4 * mb);

        Self {
            ops,
            config,
            buf_embed,
            buf_lm_head,
            buf_final_norm,
            layer_bufs,
            kv_caches,
            ws_token,
            ws_hidden,
            ws_norm,
            ws_q,
            ws_k,
            ws_v,
            ws_attn_out,
            ws_proj,
            ws_gate,
            ws_up,
            ws_ffn,
            ws_down,
            ws_residual,
            ws_logits,
            position: 0,
        }
    }

    /// Run inference: given prompt text, generate completion.
    pub fn generate(&mut self, tokenizer: &Tokenizer, prompt: &str, allowed_tokens: Option<&[u32]>) -> Result<String, String> {
        let _h = self.config.hidden_size as u32;
        let _q_h = self.config.q_heads as u32;
        let _kv_h = self.config.kv_heads as u32;
        let _hd = self.config.head_dim as u32;
        let _inter = self.config.intermediate_size as u32;
        let max_tokens = self.config.max_tokens;

        // Encode prompt
        let prompt_ids = tokenizer.encode(prompt)?;
        println!("[INFERENCE] Prompt: {} tokens", prompt_ids.len());

        let mut generated_ids: Vec<u32> = Vec::new();
        let mut all_ids = prompt_ids.clone();

        // Prefill: process all prompt tokens in batches
        let gen_start = std::time::Instant::now();
        let mut i = 0;
        while i < prompt_ids.len() {
            let end = (i + Self::MAX_BATCH).min(prompt_ids.len());
            let batch = &prompt_ids[i..end];
            self.forward_batch(batch)?;
            
            // Only need the logits from the last chunk's last token
            if end == prompt_ids.len() {
                let vocab = self.config.vocab_size;
                let logits = self.ops.read_f32_slice(&self.ws_logits, batch.len() * vocab);
                let last_logits = &logits[(batch.len() - 1) * vocab..batch.len() * vocab];
                let next_token = sample_token(last_logits, self.config.temperature);
                generated_ids.push(next_token);
                all_ids.push(next_token);
            }
            i = end;
        }

        let prefill_end = std::time::Instant::now();
        let prefill_ms = prefill_end.duration_since(gen_start).as_millis();
        println!(
            "[PERF] Prefill: {} tokens in {}ms ({:.1} tok/s)",
            prompt_ids.len(),
            prefill_ms,
            prompt_ids.len() as f64 / (prefill_ms as f64 / 1000.0)
        );

        // Autoregressive generation
        let decode_start = std::time::Instant::now();
        let mut decode_tokens = 0u32;

        for _step in 0..max_tokens.saturating_sub(1) {
            let last_token = *generated_ids.last().unwrap();

            // EOS check (model-specific)
            if last_token == self.config.eos_token_id {
                break;
            }

            let next_token = self.step(last_token, allowed_tokens)?;
            generated_ids.push(next_token);
            all_ids.push(next_token);
            decode_tokens += 1;

            // Print token as we go for streaming effect
            if let Ok(text) = tokenizer.decode(&[next_token]) {
                print!("{}", text);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
        }

        let decode_elapsed = decode_start.elapsed();
        let decode_ms = decode_elapsed.as_millis();
        let tok_per_sec = if decode_ms > 0 {
            decode_tokens as f64 / (decode_ms as f64 / 1000.0)
        } else {
            0.0
        };
        println!(
            "\n[PERF] Decode: {} tokens in {}ms ({:.1} tok/s)",
            decode_tokens, decode_ms, tok_per_sec
        );

        // Decode the full generation
        let output = tokenizer.decode(&generated_ids)?;
        Ok(output)
    }

    /// Process exactly one token (without looping context), returning the next sampled token.
    /// Used by the basic autoregressive generation loop.
    pub fn step(&mut self, token_id: u32, allowed_tokens: Option<&[u32]>) -> Result<u32, String> {
        self.forward_batch(&[token_id])?;
        
        let logits = self.read_logits(1);
        
        let next_token = if let Some(allowed) = allowed_tokens {
            let mut token_logits = logits.to_vec();
            for i in 0..token_logits.len() {
                if !allowed.contains(&(i as u32)) {
                    token_logits[i] = f32::NEG_INFINITY;
                }
            }
            sample_token(&token_logits, self.config.temperature)
        } else {
            sample_token(logits, self.config.temperature)
        };

        Ok(next_token)
    }

    /// Fork the current reasoning thread into a sub-loom.
    /// Returns the saved token position.
    pub fn push_loom(&self) -> u32 {
        self.position
    }

    /// Restore the reasoning thread from a saved sub-loom.
    /// Ephemeral tokens generated after `saved_position` will be overwritten.
    pub fn pop_loom(&mut self, saved_position: u32) {
        self.position = saved_position;
    }

    /// Read the logits buffer without copying.
    pub fn read_logits(&self, num_tokens: usize) -> &[f32] {
        let vocab = self.config.vocab_size;
        self.ops.read_f32_slice(&self.ws_logits, num_tokens * vocab)
    }

    /// Internal function to process a batch of tokens through the full transformer.
    /// Runs the ENTIRE forward pass as ONE GPU command buffer with ZERO
    /// intermediate CPU-GPU syncs.
    pub fn forward_batch(
        &mut self,
        tokens: &[u32]
    ) -> Result<(), String> {
        let num_tokens = tokens.len() as u32;
        if num_tokens == 0 || num_tokens as usize > Self::MAX_BATCH {
            return Err(format!("Batch size {} unsupported (max {})", num_tokens, Self::MAX_BATCH));
        }

        let pos = self.position;

        // ── KV cache bounds guard ────────────────────────────────────────
        if (pos + num_tokens) as usize > Self::MAX_SEQ_LEN {
            return Err(format!(
                "KV cache exhausted: position {} + {} >= MAX_SEQ_LEN {}.",
                pos, num_tokens, Self::MAX_SEQ_LEN
            ));
        }

        let h = self.config.hidden_size as u32;
        let q_h = self.config.q_heads as u32;
        let kv_h = self.config.kv_heads as u32;
        let hd = self.config.head_dim as u32;
        let inter = self.config.intermediate_size as u32;

        let total_kv_len = pos + num_tokens;
        let kv_dim = kv_h * hd;
        let kv_elem_offset = pos * kv_dim; // element offset into KV cache

        // ══════════════════════════════════════════════════════════════
        //  Batch computation: embed → layers → final norm → lm_head
        // ══════════════════════════════════════════════════════════════
        self.ops.begin_batch();

        // 1. Embedding lookup
        unsafe {
            let ptr = self.ws_token.contents() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens.len());
        }
        self.ops
            .embed_lookup(&self.ws_token, &self.buf_embed, &self.ws_hidden, h, num_tokens);

        if self.config.embedding_multiplier != 1.0 {
            self.ops
                .scale_f16(&self.ws_hidden, h * num_tokens, self.config.embedding_multiplier);
        }

        // 2. Transformer layers
        for layer_idx in 0..self.config.num_layers {
            let lb = &self.layer_bufs[layer_idx];

            std::mem::swap(&mut self.ws_hidden, &mut self.ws_residual);

            self.ops.rms_norm(
                &self.ws_hidden,
                &lb.input_layernorm,
                &self.ws_norm,
                h,
                num_tokens,
                self.config.rms_norm_eps,
            );

            self.ops
                .matmul(&self.ws_norm, &lb.q_proj, &self.ws_q, num_tokens, q_h * hd, h);
            self.ops
                .matmul(&self.ws_norm, &lb.k_proj, &self.ws_k, num_tokens, kv_dim, h);
            self.ops
                .matmul(&self.ws_norm, &lb.v_proj, &self.ws_v, num_tokens, kv_dim, h);

            if let Some(ref qb) = lb.q_bias {
                self.ops.add_bias(&self.ws_q, qb, q_h * hd, num_tokens);
            }
            if let Some(ref kb) = lb.k_bias {
                self.ops.add_bias(&self.ws_k, kb, kv_dim, num_tokens);
            }
            if let Some(ref vb) = lb.v_bias {
                self.ops.add_bias(&self.ws_v, vb, kv_dim, num_tokens);
            }

            self.ops.rope(
                &self.ws_q,
                &self.ws_k,
                num_tokens,
                q_h,
                kv_h,
                hd,
                pos,
                self.config.rope_theta,
            );

            let cache = &self.kv_caches[layer_idx];
            self.ops
                .copy_buffer(&self.ws_k, &cache.k_cache, kv_dim * num_tokens, 0, kv_elem_offset);
            self.ops
                .copy_buffer(&self.ws_v, &cache.v_cache, kv_dim * num_tokens, 0, kv_elem_offset);

            self.ops.multihead_attention(
                &self.ws_q,
                &cache.k_cache,
                &cache.v_cache,
                &self.ws_attn_out,
                total_kv_len,
                hd,
                q_h,
                kv_h,
                num_tokens
            );

            self.ops
                .matmul(&self.ws_attn_out, &lb.o_proj, &self.ws_proj, num_tokens, h, q_h * hd);

            if self.config.residual_multiplier != 1.0 {
                self.ops
                    .scale_f16(&self.ws_proj, h * num_tokens, self.config.residual_multiplier);
            }

            self.ops.add_residual(&self.ws_hidden, &self.ws_proj, h * num_tokens);
            // ws_hidden now has hidden + proj
            // ws_residual has the old hidden (which is what we want for the FFN residual)
            std::mem::swap(&mut self.ws_hidden, &mut self.ws_residual);

            self.ops.rms_norm(
                &self.ws_hidden,
                &lb.post_attention_layernorm,
                &self.ws_norm,
                h,
                num_tokens,
                self.config.rms_norm_eps,
            );

            self.ops
                .matmul(&self.ws_norm, &lb.gate_proj, &self.ws_gate, num_tokens, inter, h);
            self.ops
                .matmul(&self.ws_norm, &lb.up_proj, &self.ws_up, num_tokens, inter, h);
            self.ops
                .silu_gate(&self.ws_gate, &self.ws_up, &self.ws_ffn, inter * num_tokens);
            self.ops
                .matmul(&self.ws_ffn, &lb.down_proj, &self.ws_down, num_tokens, h, inter);

            if self.config.residual_multiplier != 1.0 {
                self.ops
                    .scale_f16(&self.ws_down, h * num_tokens, self.config.residual_multiplier);
            }

            self.ops.add_residual(&self.ws_hidden, &self.ws_down, h * num_tokens);
            
            // At the end of the layer, ensure the final result is in ws_hidden
            // It currently is, because we added the down projection to ws_hidden (which was the old ws_residual).
            // But wait, the previous block swapped ws_hidden and ws_residual, so ws_hidden is the pre-FFN state.
            // add_residual adds to ws_hidden, so ws_hidden is the post-FFN state. We are good!
        }

        // 3. Final RMSNorm + LM Head
        self.ops.rms_norm(
            &self.ws_hidden,
            &self.buf_final_norm,
            &self.ws_norm,
            h,
            num_tokens,
            self.config.rms_norm_eps,
        );

        let vocab = self.config.vocab_size as u32;
        self.ops.matmul_scaled(
            &self.ws_norm,
            &self.buf_lm_head,
            &self.ws_logits,
            num_tokens,
            vocab,
            h,
            1.0 / self.config.logits_scaling,
        );

        self.ops.end_batch();

        self.position += num_tokens;

        Ok(())
    }

    /// Reset the KV cache and position counter.
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

/// Thread-local xorshift64 PRNG state, seeded from system time.
use std::cell::Cell;
thread_local! {
    static RNG_STATE: Cell<u64> = Cell::new({
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        t.as_nanos() as u64 ^ 0xdeadbeefcafe
    });
}

fn next_random() -> f32 {
    RNG_STATE.with(|state| {
        let mut s = state.get();
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        state.set(s);
        (s % 1_000_000) as f32 / 1_000_000.0
    })
}

/// Top-p (nucleus) sampling with temperature and repetition penalty.
pub fn sample_token(logits: &[f32], temperature: f32) -> u32 {
    sample_token_advanced(logits, temperature, 0.9, &[])
}

fn sample_token_advanced(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    recent_tokens: &[u32],
) -> u32 {
    let rep_penalty = 1.1f32;

    if temperature < 0.01 {
        // Greedy: argmax
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &val) in logits.iter().enumerate() {
            let mut v = val;
            if recent_tokens.contains(&(i as u32)) {
                if v > 0.0 { v /= rep_penalty; } else { v *= rep_penalty; }
            }
            if v > max_val {
                max_val = v;
                max_idx = i as u32;
            }
        }
        return max_idx;
    }

    // Fast top-k / top-p sampling without allocating huge vectors
    const TOP_K: usize = 50;
    // Maintain top K using insertion check (O(1) fast path)
    let mut top_k = [(0usize, f32::NEG_INFINITY); TOP_K];
    
    for (i, &val) in logits.iter().enumerate() {
        let mut v = val;
        if recent_tokens.contains(&(i as u32)) {
            if v > 0.0 { v /= rep_penalty; } else { v *= rep_penalty; }
        }
        v /= temperature;
        
        if v > top_k[TOP_K - 1].1 {
            // Find insertion point
            let mut pos = TOP_K - 1;
            while pos > 0 && v > top_k[pos - 1].1 {
                pos -= 1;
            }
            
            // Shift elements down
            for j in (pos + 1..TOP_K).rev() {
                top_k[j] = top_k[j - 1];
            }
            
            top_k[pos] = (i, v);
        }
    }

    // Compute exp and softmax over just the top K
    let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(TOP_K);
    let mut sum = 0.0f32;
    let local_max = top_k[0].1;
    
    for &(idx, v) in &top_k {
        if v == f32::NEG_INFINITY || v.is_nan() { break; }
        let prob = (v - local_max).exp();
        candidates.push((idx, prob));
        sum += prob;
    }

    // Find top-p cutoff and renormalize
    let mut top_p_sum = 0.0f32;
    let mut final_candidates = Vec::with_capacity(TOP_K);
    for &(idx, prob) in &candidates {
        let normalized = prob / sum;
        final_candidates.push((idx, normalized));
        top_p_sum += normalized;
        if top_p_sum >= top_p {
            break;
        }
    }

    // Renormalize and sample
    let cand_sum: f32 = final_candidates.iter().map(|(_, p)| p).sum();
    let r = next_random();
    let mut acc = 0.0f32;
    for &(idx, prob) in &final_candidates {
        acc += prob / cand_sum;
        if acc >= r {
            return idx as u32;
        }
    }

    final_candidates.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
}
