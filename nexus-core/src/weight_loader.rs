//! Safetensors weight loader for Granite / Llama-style models.
//!
//! Reads HuggingFace safetensors files and provides zero-copy access to
//! weight tensors, converting bf16 → f16 on the fly.

use half::{bf16, f16 as F16};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// Model weight index (model.safetensors.index.json)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-layer weights
// ─────────────────────────────────────────────────────────────────────────────

/// All weight tensors for a single transformer layer (converted to f16).
pub struct LayerWeights {
    /// Query projection [hidden_size, hidden_size] 
    pub q_proj: Vec<F16>,
    /// Key projection [hidden_size, kv_dim]
    pub k_proj: Vec<F16>,
    /// Value projection [hidden_size, kv_dim]
    pub v_proj: Vec<F16>,
    /// Output projection [hidden_size, hidden_size]
    pub o_proj: Vec<F16>,
    /// Gate projection (SiLU) [hidden_size, intermediate_size]
    pub gate_proj: Vec<F16>,
    /// Up projection [hidden_size, intermediate_size]
    pub up_proj: Vec<F16>,
    /// Down projection [intermediate_size, hidden_size]
    pub down_proj: Vec<F16>,
    /// Input layernorm weights [hidden_size]
    pub input_layernorm: Vec<F16>,
    /// Post-attention layernorm weights [hidden_size]
    pub post_attention_layernorm: Vec<F16>,
    /// Optional Q/K/V biases (Qwen uses these, Granite doesn't)
    pub q_bias: Option<Vec<F16>>,
    pub k_bias: Option<Vec<F16>>,
    pub v_bias: Option<Vec<F16>>,
}

/// All model weights, loaded and converted to f16.
pub struct ModelWeights {
    /// Token embedding table [vocab_size, hidden_size]
    pub embed_tokens: Vec<F16>,
    /// Per-layer transformer weights
    pub layers: Vec<LayerWeights>,
    /// Final RMS norm weights [hidden_size]
    pub final_norm: Vec<F16>,
    /// LM head (output projection) [vocab_size, hidden_size]
    /// May be shared with embed_tokens via tie_word_embeddings
    pub lm_head: Vec<F16>,
}

// ─────────────────────────────────────────────────────────────────────────────
// bf16 → f16 conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert raw bytes (bf16 format) to a Vec of f16 values.
///
/// bf16 and f16 are both 2 bytes but have different bit layouts:
/// - bf16: 1 sign + 8 exponent + 7 mantissa (same range as f32)
/// - f16:  1 sign + 5 exponent + 10 mantissa (less range, more precision)
///
/// We go bf16 → f32 → f16 to handle the exponent/mantissa difference correctly.
fn bf16_to_f16(data: &[u8]) -> Vec<F16> {
    assert!(data.len() % 2 == 0, "bf16 data must have even byte count");
    let count = data.len() / 2;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let bytes = [data[i * 2], data[i * 2 + 1]];
        let bf = bf16::from_le_bytes(bytes);
        let f32_val = bf.to_f32();
        result.push(F16::from_f32(f32_val));
    }

    result
}

/// Convert raw bytes (already f16 format) to a Vec of f16 values.
fn f16_direct(data: &[u8]) -> Vec<F16> {
    assert!(data.len() % 2 == 0, "f16 data must have even byte count");
    let count = data.len() / 2;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let bytes = [data[i * 2], data[i * 2 + 1]];
        result.push(F16::from_le_bytes(bytes));
    }

    result
}

/// Convert raw bytes (f32 format) to a Vec of f16 values.
fn f32_to_f16(data: &[u8]) -> Vec<F16> {
    assert!(data.len() % 4 == 0, "f32 data must have byte count divisible by 4");
    let count = data.len() / 4;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
        let f = f32::from_le_bytes(bytes);
        result.push(F16::from_f32(f));
    }

    result
}

/// Convert tensor data to f16 based on its dtype.
fn tensor_to_f16(data: &[u8], dtype: Dtype) -> Result<Vec<F16>, String> {
    match dtype {
        Dtype::BF16 => Ok(bf16_to_f16(data)),
        Dtype::F16 => Ok(f16_direct(data)),
        Dtype::F32 => Ok(f32_to_f16(data)),
        other => Err(format!("Unsupported tensor dtype: {:?}", other)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight loading
// ─────────────────────────────────────────────────────────────────────────────

/// Load a tensor by name from the appropriate safetensors file.
fn load_tensor(
    tensor_name: &str,
    file_map: &HashMap<String, String>,
    file_cache: &HashMap<String, Vec<u8>>,
) -> Result<Vec<F16>, String> {
    let file_name = file_map
        .get(tensor_name)
        .ok_or_else(|| format!("Tensor '{}' not found in weight map", tensor_name))?;

    let file_data = file_cache
        .get(file_name)
        .ok_or_else(|| format!("File '{}' not loaded", file_name))?;

    let tensors = SafeTensors::deserialize(file_data)
        .map_err(|e| format!("Failed to deserialize '{}': {}", file_name, e))?;

    let tensor = tensors
        .tensor(tensor_name)
        .map_err(|e| format!("Tensor '{}' not found in '{}': {}", tensor_name, file_name, e))?;

    tensor_to_f16(tensor.data(), tensor.dtype())
}

/// Load all model weights from a safetensors model directory.
///
/// # Arguments
///
/// - `model_dir`: Path to directory containing config.json, model*.safetensors,
///   and model.safetensors.index.json
/// - `num_layers`: Number of transformer layers to load
///
/// # Returns
///
/// A `ModelWeights` containing all tensors converted to f16.
pub fn load_weights(model_dir: &str, num_layers: usize) -> Result<ModelWeights, String> {
    let model_path = Path::new(model_dir);

    println!("[WEIGHTS] Loading from '{}'...", model_dir);

    // 1. Try multi-shard index first, fall back to single-file
    let index_path = model_path.join("model.safetensors.index.json");
    let single_path = model_path.join("model.safetensors");

    let (weight_map, file_cache) = if index_path.exists() {
        // Multi-shard: read index and load each referenced file
        let index_data = fs::read_to_string(&index_path)
            .map_err(|e| format!("Failed to read index: {}", e))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_data)
            .map_err(|e| format!("Failed to parse index: {}", e))?;

        println!("[WEIGHTS] Index loaded: {} tensors across multiple shards", index.weight_map.len());

        let unique_files: Vec<String> = {
            let mut files: Vec<String> = index.weight_map.values().cloned().collect();
            files.sort();
            files.dedup();
            files
        };

        let mut cache: HashMap<String, Vec<u8>> = HashMap::new();
        for file_name in &unique_files {
            let file_path = model_path.join(file_name);
            println!("[WEIGHTS] Loading {}...", file_name);
            let data = fs::read(&file_path)
                .map_err(|e| format!("Failed to read '{}': {}", file_name, e))?;
            let size_mb = data.len() / (1024 * 1024);
            println!("[WEIGHTS]   {} MB loaded", size_mb);
            cache.insert(file_name.clone(), data);
        }

        (index.weight_map, cache)
    } else if single_path.exists() {
        // Single-shard: scan the file for all tensor names
        let file_name = "model.safetensors".to_string();
        println!("[WEIGHTS] Loading single shard: model.safetensors...");
        let data = fs::read(&single_path)
            .map_err(|e| format!("Failed to read model.safetensors: {}", e))?;
        let size_mb = data.len() / (1024 * 1024);
        println!("[WEIGHTS]   {} MB loaded", size_mb);

        // Build weight_map by scanning all tensor names in the file
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| format!("Failed to deserialize model.safetensors: {}", e))?;
        let mut wmap: HashMap<String, String> = HashMap::new();
        for name in tensors.names() {
            wmap.insert(name.to_string(), file_name.clone());
        }
        println!("[WEIGHTS]   {} tensors found in single shard", wmap.len());

        let mut cache: HashMap<String, Vec<u8>> = HashMap::new();
        cache.insert(file_name, data);

        (wmap, cache)
    } else {
        return Err(format!(
            "No model weights found in '{}'. Expected model.safetensors.index.json or model.safetensors",
            model_dir
        ));
    };

    // 2. Load embedding
    println!("[WEIGHTS] Loading embeddings...");
    let embed_tokens = load_tensor("model.embed_tokens.weight", &weight_map, &file_cache)?;
    println!("[WEIGHTS]   embed_tokens: {} values", embed_tokens.len());

    // 3. Load per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if i % 10 == 0 {
            println!("[WEIGHTS] Loading layer {}/{}...", i, num_layers);
        }
        let prefix = format!("model.layers.{}", i);

        // Load optional biases (Qwen has them, Granite does not)
        let q_bias_key = format!("{}.self_attn.q_proj.bias", prefix);
        let k_bias_key = format!("{}.self_attn.k_proj.bias", prefix);
        let v_bias_key = format!("{}.self_attn.v_proj.bias", prefix);

        let q_bias = if weight_map.contains_key(&q_bias_key) {
            Some(load_tensor(&q_bias_key, &weight_map, &file_cache)?)
        } else { None };
        let k_bias = if weight_map.contains_key(&k_bias_key) {
            Some(load_tensor(&k_bias_key, &weight_map, &file_cache)?)
        } else { None };
        let v_bias = if weight_map.contains_key(&v_bias_key) {
            Some(load_tensor(&v_bias_key, &weight_map, &file_cache)?)
        } else { None };

        if i == 0 && q_bias.is_some() {
            println!("[WEIGHTS] QKV biases detected (Qwen-style)");
        }

        let layer = LayerWeights {
            q_proj: load_tensor(&format!("{}.self_attn.q_proj.weight", prefix), &weight_map, &file_cache)?,
            k_proj: load_tensor(&format!("{}.self_attn.k_proj.weight", prefix), &weight_map, &file_cache)?,
            v_proj: load_tensor(&format!("{}.self_attn.v_proj.weight", prefix), &weight_map, &file_cache)?,
            o_proj: load_tensor(&format!("{}.self_attn.o_proj.weight", prefix), &weight_map, &file_cache)?,
            gate_proj: load_tensor(&format!("{}.mlp.gate_proj.weight", prefix), &weight_map, &file_cache)?,
            up_proj: load_tensor(&format!("{}.mlp.up_proj.weight", prefix), &weight_map, &file_cache)?,
            down_proj: load_tensor(&format!("{}.mlp.down_proj.weight", prefix), &weight_map, &file_cache)?,
            input_layernorm: load_tensor(&format!("{}.input_layernorm.weight", prefix), &weight_map, &file_cache)?,
            post_attention_layernorm: load_tensor(&format!("{}.post_attention_layernorm.weight", prefix), &weight_map, &file_cache)?,
            q_bias,
            k_bias,
            v_bias,
        };
        layers.push(layer);
    }
    println!("[WEIGHTS] All {} layers loaded", num_layers);

    // 4. Load final norm
    let final_norm = load_tensor("model.norm.weight", &weight_map, &file_cache)?;

    // 5. Load lm_head (may be tied to embed_tokens via tie_word_embeddings)
    let lm_head = if weight_map.contains_key("lm_head.weight") {
        load_tensor("lm_head.weight", &weight_map, &file_cache)?
    } else {
        println!("[WEIGHTS] lm_head tied to embed_tokens");
        embed_tokens.clone()
    };

    println!("[WEIGHTS] All weights loaded successfully");

    Ok(ModelWeights {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_to_f16_conversion() {
        // bf16 for 1.0 = 0x3F80
        let bf16_one = bf16::from_f32(1.0);
        let bytes = bf16_one.to_le_bytes();
        let data = [bytes[0], bytes[1]];
        let result = bf16_to_f16(&data);
        assert_eq!(result.len(), 1);
        let val = result[0].to_f32();
        assert!((val - 1.0).abs() < 0.01, "Expected 1.0, got {}", val);
    }

    #[test]
    fn bf16_to_f16_batch() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 42.0];
        let mut data = Vec::new();
        for &v in &values {
            let bf = bf16::from_f32(v);
            data.extend_from_slice(&bf.to_le_bytes());
        }
        let result = bf16_to_f16(&data);
        assert_eq!(result.len(), values.len());
        for (i, &expected) in values.iter().enumerate() {
            let got = result[i].to_f32();
            assert!(
                (got - expected).abs() < 0.1,
                "Index {}: expected {}, got {}", i, expected, got
            );
        }
    }
}
