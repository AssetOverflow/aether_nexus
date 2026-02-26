//! Benchmark harness for the Weaver decode kernel.
//!
//! Generates synthetic data and measures decode throughput:
//! - Wall-clock time per decode step
//! - Average latency over N iterations
//! - Estimated tokens/second
//!
//! Run with: `cargo run -p nexus-core -- --bench`

use crate::types::{SparseCode, WeaverParams};
use crate::weaver::WeaverEngine;
use half::f16 as F16;
use std::time::Instant;

/// Configuration for the benchmark
pub struct BenchConfig {
    /// Number of decode iterations
    pub iterations: usize,
    /// Number of hot blocks
    pub hot_blocks: u32,
    /// Number of cold blocks
    pub cold_blocks: u32,
    /// Number of warmup iterations (excluded from timing)
    pub warmup: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            hot_blocks: 64,
            cold_blocks: 16,
            warmup: 10,
        }
    }
}

/// Results from a benchmark run
pub struct BenchResults {
    /// Total wall-clock time for all iterations (excluding warmup)
    pub total_time_ms: f64,
    /// Average latency per decode step in milliseconds
    pub avg_latency_ms: f64,
    /// Estimated tokens per second
    pub tokens_per_sec: f64,
    /// Number of iterations measured
    pub iterations: usize,
    /// Total KV tokens attended to per step
    pub kv_tokens: usize,
    /// Device name
    pub device_name: String,
}

impl std::fmt::Display for BenchResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════╗")?;
        writeln!(f, "║       Weaver Kernel Benchmark Results            ║")?;
        writeln!(f, "╚══════════════════════════════════════════════════╝")?;
        writeln!(f)?;
        writeln!(f, "  Device:           {}", self.device_name)?;
        writeln!(f, "  Iterations:       {}", self.iterations)?;
        writeln!(f, "  KV tokens/step:   {}", self.kv_tokens)?;
        writeln!(f)?;
        writeln!(f, "  Total time:       {:.2} ms", self.total_time_ms)?;
        writeln!(f, "  Avg latency:      {:.3} ms/token", self.avg_latency_ms)?;
        writeln!(f, "  Throughput:       {:.1} tok/s", self.tokens_per_sec)?;
        writeln!(f)?;

        // Compare against Yellowpaper targets
        if self.tokens_per_sec >= 92.0 {
            writeln!(f, "  ✅ Meets Yellowpaper target (≥92 tok/s)")?;
        } else {
            writeln!(f, "  ⚠️  Below Yellowpaper target (92 tok/s). This is expected")?;
            writeln!(f, "      with synthetic data and unoptimized debug builds.")?;
        }
        Ok(())
    }
}

/// Run the Weaver kernel benchmark.
///
/// This generates synthetic data of the correct shape and measures
/// GPU decode throughput.
pub fn run_benchmark(
    engine: &mut WeaverEngine,
    config: &BenchConfig,
) -> Result<BenchResults, String> {
    let q_heads: u32 = 32;
    let kv_heads: u32 = 8;
    let head_dim: u32 = 128;
    let dict_size: u32 = 512;
    let block_size: u32 = 16;
    let hot_count = config.hot_blocks;
    let cold_count = config.cold_blocks;

    // Total KV tokens attended to per decode step
    let kv_tokens = ((hot_count + cold_count) * block_size) as usize;

    println!("[BENCH] Generating synthetic data...");
    println!("[BENCH]   q_heads={}, kv_heads={}, head_dim={}", q_heads, kv_heads, head_dim);
    println!("[BENCH]   hot_blocks={}, cold_blocks={}, block_size={}", hot_count, cold_count, block_size);
    println!("[BENCH]   KV tokens/step: {}", kv_tokens);

    // Generate synthetic data with small random-ish values
    let val = F16::from_f32(0.01);
    let q_exact = vec![val; (q_heads * head_dim) as usize];
    let q_latent = vec![val; (q_heads * dict_size) as usize];
    let hot_pool = vec![val; (hot_count * block_size * kv_heads * head_dim) as usize];
    let cold_pool = vec![SparseCode::zero(); (cold_count * block_size * kv_heads) as usize];
    let loom_refs: Vec<u32> = (0..(hot_count + cold_count)).collect();
    let dictionary = vec![val; (kv_heads * dict_size * head_dim) as usize];

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

    // Warmup
    println!("[BENCH] Warmup ({} iterations)...", config.warmup);
    for _ in 0..config.warmup {
        engine.decode(
            &q_exact, &q_latent, &hot_pool, &cold_pool,
            &loom_refs, &dictionary, &params,
        )?;
    }

    // Benchmark
    println!("[BENCH] Running {} iterations...", config.iterations);
    let start = Instant::now();

    for _ in 0..config.iterations {
        engine.decode(
            &q_exact, &q_latent, &hot_pool, &cold_pool,
            &loom_refs, &dictionary, &params,
        )?;
    }

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_ms = total_ms / config.iterations as f64;
    let tok_s = 1000.0 / avg_ms;

    Ok(BenchResults {
        total_time_ms: total_ms,
        avg_latency_ms: avg_ms,
        tokens_per_sec: tok_s,
        iterations: config.iterations,
        kv_tokens,
        device_name: engine.device_name().to_string(),
    })
}
