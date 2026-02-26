//! Inference engine for Granite / Llama-style transformers.
//!
//! Implements the full autoregressive decode loop:
//!   embed → (RMSNorm → Attn → Residual → RMSNorm → FFN → Residual) × N → Logits → Sample

use crate::ops::OpsEngine;
use crate::weight_loader::ModelWeights;
use crate::tokenizer::Tokenizer;
use half::f16 as F16;
use metal::Buffer;

/// Configuration for the inference engine.
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
            attention_multiplier: 0.015625,  // 1/64
            residual_multiplier: 0.22,
            logits_scaling: 8.0,
            temperature: 0.7,
            max_tokens: 128,
            eos_token_id: 0,
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
            head_dim: 64,       // 896 / 14 = 64
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
            eos_token_id: 151645,  // <|im_end|>
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
            head_dim: 128,      // 1536 / 12 = 128
            intermediate_size: 8960,
            vocab_size: 151936,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            embedding_multiplier: 1.0,
            attention_multiplier: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            temperature: 0.6,   // Slightly lower for more focused reasoning
            max_tokens: 512,    // Longer context for reasoning chains
            eos_token_id: 151643,  // DeepSeek uses bos_token_id as eos
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

        let hidden_size = config["hidden_size"].as_u64().unwrap_or(0) as usize;
        let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(0) as usize;

        // Match by (hidden_size, num_layers) fingerprint
        match (hidden_size, num_layers) {
            (2048, 40) => {
                println!("[DETECT] Matched: Granite 3.0 2B Instruct");
                Ok(Self::granite_2b())
            }
            (896, 24) => {
                println!("[DETECT] Matched: Qwen 2.5 0.5B Instruct");
                Ok(Self::qwen_0_5b())
            }
            (1536, 28) => {
                println!("[DETECT] Matched: DeepSeek-R1-Distill-Qwen-1.5B");
                Ok(Self::deepseek_r1_1_5b())
            }
            _ => Err(format!(
                "Unknown model architecture: hidden_size={}, num_layers={}. Add a config preset.",
                hidden_size, num_layers
            )),
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
    ws_q_head: Buffer,
    ws_head_out: Buffer,
    ws_k_head_cache: Buffer,
    ws_v_head_cache: Buffer,
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
    /// Create an inference engine, uploading all weights to GPU.
    pub fn new(
        ops: OpsEngine,
        weights: &ModelWeights,
        config: InferenceConfig,
    ) -> Self {
        println!("[INFERENCE] Uploading weights to GPU...");

        let buf_embed = ops.buffer_f16(&weights.embed_tokens);
        let buf_lm_head = ops.buffer_f16(&weights.lm_head);
        let buf_final_norm = ops.buffer_f16(&weights.final_norm);

        let mut layer_bufs = Vec::with_capacity(config.num_layers);
        for (i, layer) in weights.layers.iter().enumerate() {
            if i % 10 == 0 {
                println!("[INFERENCE]   Uploading layer {}/{}...", i, config.num_layers);
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

        // Allocate KV caches (max 4096 tokens)
        let max_seq = 4096u64;
        let kv_dim = (config.kv_heads * config.head_dim) as u64;
        let cache_size = max_seq * kv_dim * 2; // f16 = 2 bytes

        let mut kv_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            kv_caches.push(KvCache {
                k_cache: ops.buffer_empty(cache_size),
                v_cache: ops.buffer_empty(cache_size),
            });
        }

        println!("[INFERENCE] All weights uploaded. KV cache allocated for {} tokens.", max_seq);

        // Workspace sizes
        let h = config.hidden_size as u64;
        let q_h = config.q_heads as u64;
        let kv_h = config.kv_heads as u64;
        let hd = config.head_dim as u64;
        let inter = config.intermediate_size as u64;
        let vocab = config.vocab_size as u64;

        let ws_hidden = ops.buffer_empty(h * 2);
        let ws_norm = ops.buffer_empty(h * 2);
        let ws_q = ops.buffer_empty(q_h * hd * 2);
        let ws_k = ops.buffer_empty(kv_dim * 2);
        let ws_v = ops.buffer_empty(kv_dim * 2);
        let ws_attn_out = ops.buffer_empty(q_h * hd * 2);
        let ws_proj = ops.buffer_empty(h * 2);
        let ws_gate = ops.buffer_empty(inter * 2);
        let ws_up = ops.buffer_empty(inter * 2);
        let ws_ffn = ops.buffer_empty(inter * 2);
        let ws_down = ops.buffer_empty(h * 2);
        let ws_residual = ops.buffer_empty(h * 2);
        let ws_q_head = ops.buffer_empty(hd * 2);
        let ws_head_out = ops.buffer_empty(hd * 2);
        let ws_k_head_cache = ops.buffer_empty(max_seq * hd * 2);
        let ws_v_head_cache = ops.buffer_empty(max_seq * hd * 2);
        let ws_logits = ops.buffer_empty(vocab * 4);

        Self {
            ops,
            config,
            buf_embed,
            buf_lm_head,
            buf_final_norm,
            layer_bufs,
            kv_caches,
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
            ws_q_head,
            ws_head_out,
            ws_k_head_cache,
            ws_v_head_cache,
            ws_logits,
            position: 0,
        }
    }

    /// Run inference: given prompt text, generate completion.
    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
    ) -> Result<String, String> {
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

        // Prefill: process all prompt tokens
        let gen_start = std::time::Instant::now();
        for (i, &token_id) in prompt_ids.iter().enumerate() {
            let next_token = self.step(token_id)?;
            // During prefill, we don't use the output until the last token
            if i == prompt_ids.len() - 1 {
                // Sample from the logits of the last prompt token
                generated_ids.push(next_token);
                all_ids.push(next_token);
            }
        }

        let prefill_end = std::time::Instant::now();
        let prefill_ms = prefill_end.duration_since(gen_start).as_millis();
        println!("[PERF] Prefill: {} tokens in {}ms ({:.1} tok/s)", 
            prompt_ids.len(), prefill_ms, 
            prompt_ids.len() as f64 / (prefill_ms as f64 / 1000.0));

        // Autoregressive generation
        let decode_start = std::time::Instant::now();
        let mut decode_tokens = 0u32;

        for step in 0..max_tokens.saturating_sub(1) {
            let last_token = *generated_ids.last().unwrap();
            
            // EOS check (model-specific)
            if last_token == self.config.eos_token_id {
                break;
            }

            let next_token = self.step(last_token)?;
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
        } else { 0.0 };
        println!("\n[PERF] Decode: {} tokens in {}ms ({:.1} tok/s)", 
            decode_tokens, decode_ms, tok_per_sec);

        // Decode the full generation
        let output = tokenizer.decode(&generated_ids)?;
        Ok(output)
    }

    /// Process exactly one token (without looping context), returning the next sampled token.
    /// This gives external agent loops control over the generation pipeline.
    pub fn step(&mut self, token_id: u32) -> Result<u32, String> {
        let h = self.config.hidden_size as u32;
        let q_h = self.config.q_heads as u32;
        let kv_h = self.config.kv_heads as u32;
        let hd = self.config.head_dim as u32;
        let inter = self.config.intermediate_size as u32;
        self.forward_one_token(token_id, h, q_h, kv_h, hd, inter)
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

    /// Query the NeuralWiki (Cold Pool) for sparse blocks matching the current hidden state.
    /// Returns the indices of the top-K most relevant distilled memory blocks.
    pub fn retrieve_memory(&self, _query_hidden_state: &[half::f16], _top_k: usize) -> Vec<usize> {
        // In the true Weaver architecture, this is dispatched as a sparse-dense matmul kernel
        // For CPU simulation, we just scan the SparseCodes in the cold pool.
        
        // Return mock indices for now to validate agent wiring without the full GPU indexer
        let mut results = Vec::new();
        if self.config.max_tokens > 0 {
            results.push(0); 
        }
        results
    }
    /// Internal function to process one token through the full transformer.
    /// Runs the ENTIRE forward pass as ONE GPU command buffer with ZERO
    /// intermediate CPU-GPU syncs (for Qwen — Granite may need sync for multipliers).
    fn forward_one_token(
        &mut self,
        token_id: u32,
        h: u32, q_h: u32, kv_h: u32, hd: u32, inter: u32,
    ) -> Result<u32, String> {
        let pos = self.position;
        let kv_len = pos + 1;
        let kv_dim = kv_h * hd;
        let kv_elem_offset = pos * kv_dim; // element offset into KV cache (f16 elements)

        // ══════════════════════════════════════════════════════════════
        //  Single batch: embed → 24 layers → final norm → lm_head
        // ══════════════════════════════════════════════════════════════
        self.ops.begin_batch();

        // 1. Embedding lookup
        let token_buf = self.ops.buffer_u32(&[token_id]);
        self.ops.embed_lookup(&token_buf, &self.buf_embed, &self.ws_hidden, h, 1);

        // Embedding multiplier (Granite-specific — skip for Qwen where multiplier == 1.0)
        if self.config.embedding_multiplier != 1.0 {
            self.ops.end_batch();
            let hidden_data = self.ops.read_f16(&self.ws_hidden, h as usize);
            let scaled: Vec<F16> = hidden_data.iter()
                .map(|v| F16::from_f32(v.to_f32() * self.config.embedding_multiplier))
                .collect();
            let scaled_buf = self.ops.buffer_f16(&scaled);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scaled_buf.contents() as *const u8,
                    self.ws_hidden.contents() as *mut u8,
                    h as usize * 2,
                );
            }
            self.ops.begin_batch();
        }

        // 2. Transformer layers — ALL on GPU, no CPU round-trips
        for layer_idx in 0..self.config.num_layers {
            let lb = &self.layer_bufs[layer_idx];

            // Save residual (GPU copy)
            self.ops.copy_buffer(&self.ws_hidden, &self.ws_residual, h, 0, 0);

            // Input LayerNorm
            self.ops.rms_norm(
                &self.ws_hidden, &lb.input_layernorm, &self.ws_norm,
                h, 1, self.config.rms_norm_eps,
            );

            // Q/K/V projections
            self.ops.matmul(&self.ws_norm, &lb.q_proj, &self.ws_q, 1, q_h * hd, h);
            self.ops.matmul(&self.ws_norm, &lb.k_proj, &self.ws_k, 1, kv_dim, h);
            self.ops.matmul(&self.ws_norm, &lb.v_proj, &self.ws_v, 1, kv_dim, h);

            // QKV biases (Qwen-style)
            if let Some(ref qb) = lb.q_bias {
                self.ops.add_residual(&self.ws_q, qb, q_h * hd);
            }
            if let Some(ref kb) = lb.k_bias {
                self.ops.add_residual(&self.ws_k, kb, kv_dim);
            }
            if let Some(ref vb) = lb.v_bias {
                self.ops.add_residual(&self.ws_v, vb, kv_dim);
            }

            // RoPE
            self.ops.rope(&self.ws_q, &self.ws_k, 1, q_h, kv_h, hd, pos, self.config.rope_theta);

            // Store K/V into cache (GPU copy — no CPU sync needed!)
            let cache = &self.kv_caches[layer_idx];
            self.ops.copy_buffer(&self.ws_k, &cache.k_cache, kv_dim, 0, kv_elem_offset);
            self.ops.copy_buffer(&self.ws_v, &cache.v_cache, kv_dim, 0, kv_elem_offset);

            // Multi-head GQA attention
            self.ops.multihead_attention(
                &self.ws_q, &cache.k_cache, &cache.v_cache, &self.ws_attn_out,
                kv_len, hd, q_h, kv_h,
            );

            // Output projection
            self.ops.matmul(&self.ws_attn_out, &lb.o_proj, &self.ws_proj, 1, h, q_h * hd);

            // Residual multiplier (Granite-specific)
            if self.config.residual_multiplier != 1.0 {
                self.ops.end_batch();
                let proj_data = self.ops.read_f16(&self.ws_proj, h as usize);
                let scaled: Vec<F16> = proj_data.iter()
                    .map(|v| F16::from_f32(v.to_f32() * self.config.residual_multiplier))
                    .collect();
                let scaled_buf = self.ops.buffer_f16(&scaled);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        scaled_buf.contents() as *const u8,
                        self.ws_proj.contents() as *mut u8,
                        h as usize * 2,
                    );
                }
                self.ops.begin_batch();
            }

            // hidden = residual + proj (add_residual is in-place: ws_hidden += ws_proj)
            // But ws_hidden was overwritten by rms_norm input! We saved it in ws_residual.
            // Copy residual back to ws_hidden first, then add.
            self.ops.copy_buffer(&self.ws_residual, &self.ws_hidden, h, 0, 0);
            self.ops.add_residual(&self.ws_hidden, &self.ws_proj, h);

            // Save residual for FFN (GPU copy)
            self.ops.copy_buffer(&self.ws_hidden, &self.ws_residual, h, 0, 0);

            // Post-attention LayerNorm
            self.ops.rms_norm(
                &self.ws_hidden, &lb.post_attention_layernorm, &self.ws_norm,
                h, 1, self.config.rms_norm_eps,
            );

            // FFN
            self.ops.matmul(&self.ws_norm, &lb.gate_proj, &self.ws_gate, 1, inter, h);
            self.ops.matmul(&self.ws_norm, &lb.up_proj, &self.ws_up, 1, inter, h);
            self.ops.silu_gate(&self.ws_gate, &self.ws_up, &self.ws_ffn, inter);
            self.ops.matmul(&self.ws_ffn, &lb.down_proj, &self.ws_down, 1, h, inter);

            // Residual multiplier (Granite-specific)
            if self.config.residual_multiplier != 1.0 {
                self.ops.end_batch();
                let down_data = self.ops.read_f16(&self.ws_down, h as usize);
                let scaled: Vec<F16> = down_data.iter()
                    .map(|v| F16::from_f32(v.to_f32() * self.config.residual_multiplier))
                    .collect();
                let scaled_buf = self.ops.buffer_f16(&scaled);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        scaled_buf.contents() as *const u8,
                        self.ws_down.contents() as *mut u8,
                        h as usize * 2,
                    );
                }
                self.ops.begin_batch();
            }

            // hidden = residual + down
            self.ops.copy_buffer(&self.ws_residual, &self.ws_hidden, h, 0, 0);
            self.ops.add_residual(&self.ws_hidden, &self.ws_down, h);
        }

        // 3. Final RMSNorm + LM Head
        self.ops.rms_norm(
            &self.ws_hidden, &self.buf_final_norm, &self.ws_norm,
            h, 1, self.config.rms_norm_eps,
        );

        let vocab = self.config.vocab_size as u32;
        self.ops.matmul_scaled(
            &self.ws_norm, &self.buf_lm_head, &self.ws_logits,
            1, vocab, h, 1.0 / self.config.logits_scaling,
        );

        // ══════════════════════════════════════════════════════════════
        //  Single GPU sync — read logits from GPU
        // ══════════════════════════════════════════════════════════════
        self.ops.end_batch();

        let logits = self.ops.read_f32(&self.ws_logits, vocab as usize);
        let next_token = sample_token(&logits, self.config.temperature);

        self.position += 1;

        Ok(next_token)
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
fn sample_token(logits: &[f32], temperature: f32) -> u32 {
    sample_token_advanced(logits, temperature, 0.9, &[])
}

fn sample_token_advanced(
    logits: &[f32], temperature: f32, top_p: f32, recent_tokens: &[u32],
) -> u32 {
    let mut logits = logits.to_vec();

    // Apply repetition penalty (1.1x for recently used tokens)
    let rep_penalty = 1.1f32;
    for &tok in recent_tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= rep_penalty;
            } else {
                logits[idx] *= rep_penalty;
            }
        }
    }

    if temperature < 0.01 {
        // Greedy: argmax
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

    // Sort by probability (descending) for top-p selection
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find top-p cutoff
    let mut cumulative = 0.0f32;
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for &(idx, prob) in &indexed {
        candidates.push((idx, prob));
        cumulative += prob;
        if cumulative >= top_p {
            break;
        }
    }

    // Renormalize and sample
    let cand_sum: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r = next_random();
    let mut acc = 0.0f32;
    for &(idx, prob) in &candidates {
        acc += prob / cand_sum;
        if acc >= r {
            return idx as u32;
        }
    }

    candidates.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
}

