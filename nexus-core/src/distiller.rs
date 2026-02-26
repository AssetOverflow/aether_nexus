//! ANE Distiller – Background memory consolidation (REM cycle)
//!
//! Runs as an async background task, periodically evaluating the entropy
//! of hot KV blocks and distilling low-entropy blocks into SparseCode
//! representations using the learned dictionary.
//!
//! # Yellowpaper Invariants
//!
//! - Asymmetric pipeline: GPU for matmul projection, GPU for top-k + packing
//! - Milliwatt operation (does not compete with decode for GPU cycles)
//! - Zero-copy writes into the cold pool via Fabric mutable accessors
//! - Tunable REM interval and entropy threshold
//!
//! # Note on ANE
//!
//! The Yellowpaper specifies ANE for entropy evaluation. The real mlx-rs 0.21
//! does not expose `Device::ANE` — all operations run on the default Metal GPU.
//! When mlx-rs gains ANE support, the `evaluate_entropy` method can be trivially
//! updated to pin the computation to the ANE.

use crate::fabric::Fabric;
use crate::types::{MemoryConfig, ModelDims, SparseCode};

use half::f16;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, sleep};

// ─────────────────────────────────────────────────────────────────────────────
// Distiller
// ─────────────────────────────────────────────────────────────────────────────

/// The ANE Distiller – the organism's REM sleep.
///
/// Runs in the background, evaluating the entropy of hot KV blocks and
/// distilling low-entropy (repetitive, predictable) blocks into compact
/// SparseCode representations.
pub struct Distiller<D: ModelDims> {
    config: MemoryConfig,
    _dims: std::marker::PhantomData<D>,
}

impl<D: ModelDims> Distiller<D> {
    /// Create a new Distiller.
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            _dims: std::marker::PhantomData,
        }
    }

    /// Run the REM cycle – evaluates entropy and distills blocks.
    ///
    /// This is designed to be called periodically (every REM_INTERVAL_SECS).
    /// In a full deployment it runs as a `tokio::spawn` background task.
    ///
    /// # Arguments
    ///
    /// - `fabric`: Shared mutable reference to the Fabric (behind a Mutex)
    pub async fn run(self, fabric: Arc<Mutex<Fabric<D>>>) {
        loop {
            sleep(Duration::from_secs(self.config.rem_interval_secs)).await;

            // 1. Evaluate entropy of hot blocks
            let candidates = {
                let fab = fabric.lock().await;
                self.evaluate_entropy(&fab)
            };

            if !candidates.is_empty() {
                // 2. Distill candidates (locking individually to allow inference to interleave)
                for block_idx in candidates {
                    let mut fab = fabric.lock().await;
                    if let Err(e) = Self::distill_block(&mut fab, block_idx) {
                        eprintln!("Distiller: block {} failed: {}", block_idx, e);
                    }
                }
            }

            // 3. Persist
            {
                let mut fab: tokio::sync::MutexGuard<'_, Fabric<D>> = fabric.lock().await;
                if let Err(e) = fab.checkpoint() {
                    eprintln!("Distiller: checkpoint failed: {}", e);
                }
            }
        }
    }

    /// Evaluate entropy of hot blocks.
    ///
    /// Computes a simple variance-based entropy proxy for each hot block.
    /// Returns indices of blocks below the distillation threshold.
    ///
    /// In the full implementation, this would use MLX matmul on the ANE.
    /// For now, we compute directly from the mmap'd bytes.
    fn evaluate_entropy(&self, fabric: &Fabric<D>) -> Vec<usize> {
        let mut candidates = Vec::new();
        // Determine number of active hot blocks
        // Using radix metadata (assuming radix block 0 maps to hot block 0)
        let num_blocks =
            fabric.regions.hot_pool_size / (D::BLOCK_SIZE * D::KV_HEADS * D::HEAD_DIM * 2);

        let block_byte_size = D::BLOCK_SIZE * D::KV_HEADS * D::HEAD_DIM * 2;
        let hot_pool = fabric.hot_pool();

        for block_idx in 0..num_blocks.min(8192) {
            let offset = block_idx * block_byte_size;
            let block_data = &hot_pool[offset..offset + block_byte_size];

            // Heuristic ANE computation
            let entropy = Self::compute_block_entropy(block_data);

            // Distill if entropy is low (but > 0 to ignore empty zero blocks)
            if entropy < self.config.distill_entropy_threshold && entropy > 0.0 {
                candidates.push(block_idx);
            }
        }
        candidates
    }

    /// Compute a simple entropy proxy for a block of f16 data.
    ///
    /// Uses variance of the raw byte values as an approximation.
    /// Low variance → low entropy → good candidate for sparse compression.
    fn compute_block_entropy(block_bytes: &[u8]) -> f32 {
        if block_bytes.is_empty() {
            return 0.0;
        }

        // Interpret as f16 values
        let f16_values: &[f16] = bytemuck::cast_slice(block_bytes);

        if f16_values.is_empty() {
            return 0.0;
        }

        // All zeros = definitely low entropy but skip (empty block)
        let all_zero = f16_values.iter().all(|&v| f32::from(v) == 0.0);
        if all_zero {
            return 0.0;
        }

        // Compute mean
        let sum: f64 = f16_values.iter().map(|&v| f64::from(f32::from(v))).sum();
        let mean = sum / f16_values.len() as f64;

        // Compute variance
        let var: f64 = f16_values
            .iter()
            .map(|&v| {
                let diff = f64::from(f32::from(v)) - mean;
                diff * diff
            })
            .sum::<f64>()
            / f16_values.len() as f64;

        var as f32
    }

    /// Distill a single hot block into SparseCode representation.
    ///
    /// In the full implementation, this performs:
    /// 1. Dictionary projection (matmul on GPU): block × dictionary^T
    /// 2. Top-K selection (sort + gather): select K highest-scoring dict vectors
    /// 3. Coefficient computation: solve sparse approximation
    /// 4. Pack into SparseCode and write to cold pool
    ///
    /// For now, we create a zeroed SparseCode (placeholder for the actual
    /// dictionary projection).
    fn distill_block(
        fabric: &mut Fabric<D>,
        block_idx: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let codes_per_block = D::BLOCK_SIZE * D::KV_HEADS;
        let code_size = std::mem::size_of::<SparseCode>();
        let cold_offset = block_idx * codes_per_block * code_size;

        if cold_offset + codes_per_block * code_size > fabric.regions.cold_pool_size {
            return Err("cold pool overflow".into());
        }

        let head_dim = D::HEAD_DIM;
        let dict_size = D::DICT_SIZE;

        // Perform dictionary projection and top-K selection for each token & head
        // Note: CPU bound for now, in a full release this is an ANE/GPU kernel
        let mut sparse_codes = Vec::with_capacity(codes_per_block);

        // We read from the view before taking mutable access to the fabric's cold pool
        let hot_bytes = fabric.hot_pool();
        let dict_bytes = fabric.dictionary();

        let hot_base = block_idx * codes_per_block * head_dim * 2;

        for i in 0..codes_per_block {
            let token_offset = hot_base + i * head_dim * 2;
            let token_slice = &hot_bytes[token_offset..token_offset + head_dim * 2];
            let token_f16: &[f16] = bytemuck::cast_slice(token_slice);

            let head_idx = i % D::KV_HEADS;
            let dict_head_offset = head_idx * dict_size * head_dim * 2;

            // 1. Compute dot products with all dictionary vectors for this head
            let mut scores: Vec<(usize, f32)> = Vec::with_capacity(dict_size);
            for d in 0..dict_size {
                let d_off = dict_head_offset + d * head_dim * 2;
                // If dictionary is empty/zeroes (e.g. genesis), this will naturally yield 0.0
                let dict_vec: &[f16] =
                    bytemuck::cast_slice(&dict_bytes[d_off..d_off + head_dim * 2]);

                let dot: f32 = token_f16
                    .iter()
                    .zip(dict_vec.iter())
                    .map(|(a, b)| f32::from(*a) * f32::from(*b))
                    .sum();
                scores.push((d, dot));
            }

            // 2. Top-K selection (K = 4)
            scores.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut code = SparseCode::zero();
            for k in 0..4 {
                if k < scores.len() {
                    // Safety check based on previous test failing: index bounding
                    let idx = scores[k].0;
                    code.indices[k] = idx as u16;
                    code.coeffs[k] = f16::from_f32(scores[k].1);
                }
            }
            sparse_codes.push(code);
        }

        // Write the packed SparseCodes to the cold pool
        let cold = fabric.cold_pool_mut();
        for (i, code) in sparse_codes.iter().enumerate() {
            let offset = cold_offset + i * code_size;
            let code_bytes = bytemuck::bytes_of(code);
            cold[offset..offset + code_size].copy_from_slice(code_bytes);
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::create_genesis;
    use crate::types::Llama8B;
    use ring::rand::SystemRandom;
    use ring::signature::Ed25519KeyPair;

    #[test]
    fn entropy_computation_zero_block() {
        let zeros = vec![0u8; 1024];
        let entropy = Distiller::<Llama8B>::compute_block_entropy(&zeros);
        assert_eq!(entropy, 0.0, "all-zero block should have zero entropy");
    }

    #[test]
    fn entropy_computation_uniform_block() {
        // Create a block of uniform non-zero f16 values
        let val = f16::from_f32(1.0);
        let val_bytes = val.to_le_bytes();
        let mut block = Vec::new();
        for _ in 0..512 {
            block.extend_from_slice(&val_bytes);
        }
        let entropy = Distiller::<Llama8B>::compute_block_entropy(&block);
        assert!(
            entropy < 0.001,
            "uniform block should have very low entropy, got {}",
            entropy
        );
    }

    #[test]
    fn distill_block_writes_to_cold_pool() {
        let tmp = std::env::temp_dir().join("test_distill.aether");
        let rng = SystemRandom::new();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();

        create_genesis::<Llama8B>(tmp.to_str().unwrap(), &key_pair).unwrap();
        let mut fabric = Fabric::<Llama8B>::boot(tmp.to_str().unwrap()).unwrap();

        // Distill block 0
        Distiller::<Llama8B>::distill_block(&mut fabric, 0).unwrap();

        // Verify cold pool has SparseCode written
        let codes = fabric.cold_pool_codes();
        let first = codes[0].indices;
        // With a zeroed dictionary (genesis), the first 4 indices (0,1,2,3)
        // will have equal 0.0 scores and be selected as top-K.
        assert_eq!(
            first,
            [0, 1, 2, 3],
            "distilled code should have top-4 indices"
        );

        let _ = std::fs::remove_file(&tmp);
    }
}
