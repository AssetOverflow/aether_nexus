//
// weaver_kernel.metal
//
// AetherNexus – Unified Weaver Decode Kernel
//
// Implements the Yellowpaper v1.3 decode algorithm:
//   - Pre-segmented hot/cold passes (divergence-free)
//   - Online softmax (FlashAttention-3 style)
//   - Decompression-free sparse attention for cold blocks:
//     Q·K^T ≈ (Q·D)·C  where D = dictionary, C = sparse coefficients
//   - GQA stride support (Q_HEADS / KV_HEADS grouping)
//
// Bound to the Cortex via build.rs → metallib → runtime load
//

#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// Constants (must match ModelDims in types.rs)
// ─────────────────────────────────────────────────────────────────────────────

constant constexpr uint HEAD_DIM    = 128;
constant constexpr uint BLOCK_SIZE  = 16;
constant constexpr uint SPARSITY_K  = 4;
constant constexpr uint DICT_SIZE   = 512;

// ─────────────────────────────────────────────────────────────────────────────
// Data Types
// ─────────────────────────────────────────────────────────────────────────────

/// SparseCode: exactly 16 bytes (128-bit aligned load)
/// Must match #[repr(C, packed)] SparseCode in types.rs
struct SparseCode {
    ushort indices[4];    // dictionary indices (0..511)
    half   coeffs[4];     // sparse coefficients
};

/// Kernel dispatch parameters (from WeaverParams in types.rs)
struct WeaverParams {
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    uint block_size;
    uint gqa_group;
    uint hot_count;
    uint cold_count;
    uint dict_size;
    uint sparsity_k;
    uint _pad[3];
};

// ─────────────────────────────────────────────────────────────────────────────
// Online Softmax Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// SIMD warp-level reduction for a single float value (32-wide)
inline float simd_reduce_add(float val, uint tid) {
    // Metal SIMD shuffle reduction across 32 threads
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

/// Online softmax update: merge a new score into running max/sum/accumulator
inline void online_softmax_update(
    float score,
    thread float& m,       // running max
    thread float& l,       // running sum of exp
    thread float* acc,     // running weighted accumulator (HEAD_DIM/32 elements per thread)
    const device half* value_row,
    uint head_dim_chunk,
    uint tid)
{
    float new_m = max(m, score);
    float exp_old = exp(m - new_m);
    float exp_new = exp(score - new_m);

    // Rescale existing accumulator
    for (uint i = 0; i < head_dim_chunk; i++) {
        acc[i] *= exp_old;
    }
    l = l * exp_old + exp_new;
    m = new_m;

    // Accumulate new value contribution
    for (uint i = 0; i < head_dim_chunk; i++) {
        uint idx = tid + i * 32;
        if (idx < HEAD_DIM) {
            acc[i] += exp_new * float(value_row[idx]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Weaver Decode Kernel
// ─────────────────────────────────────────────────────────────────────────────

/// Unified decode kernel implementing pre-segmented hot/cold attention.
///
/// Grid layout:
///   - gid.x: batch (usually 1 for single-sequence decode)
///   - gid.y: query head index (0..q_heads-1)
///   - threadgroup: 32 threads (one SIMD group)
///
/// Buffer bindings:
///   0: q_exact    - query vectors for hot path [q_heads, head_dim] f16
///   1: q_latent   - query projected into dictionary space for cold path [q_heads, dict_size] f16
///   2: hot_pool   - exact f16 KV blocks [n_hot_blocks, block_size, kv_heads, head_dim] f16
///   3: cold_pool  - sparse dictionary codes [n_cold_blocks, block_size, kv_heads] SparseCode
///   4: loom_refs  - block reference indices for current persona [hot_count + cold_count] u32
///   5: output     - output token [q_heads, head_dim] f16
///   6: params     - WeaverParams constant buffer
///   7: dictionary - learned dictionary [kv_heads, dict_size, head_dim] f16
kernel void weaver_decode(
    const device half*        q_exact    [[buffer(0)]],
    const device half*        q_latent   [[buffer(1)]],
    const device half*        hot_pool   [[buffer(2)]],
    const device SparseCode*  cold_pool  [[buffer(3)]],
    const device uint*        loom_refs  [[buffer(4)]],
    device half*              output     [[buffer(5)]],
    constant WeaverParams&    p          [[buffer(6)]],
    const device half*        dictionary [[buffer(7)]],
    uint3 gid                            [[thread_position_in_grid]],
    uint  tid                            [[thread_index_in_threadgroup]])
{
    uint head = gid.y;
    uint kv_head = head / p.gqa_group;

    // Number of head_dim elements this thread is responsible for
    uint head_dim_chunk = (HEAD_DIM + 31) / 32;

    // Online softmax state (per-thread)
    thread float m = -INFINITY;
    thread float l = 0.0f;
    thread float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // HEAD_DIM/32 = 4

    // ─── Pass 1: Hot segment (exact f16 attention) ───────────────────

    for (uint b = 0; b < p.hot_count; b++) {
        uint block_ref = loom_refs[b];
        uint phys_block = block_ref & 0x7FFFFFFF;

        for (uint t = 0; t < BLOCK_SIZE; t++) {
            // Compute exact dot product: score = Q · K
            uint k_offset = phys_block * BLOCK_SIZE * p.kv_heads * HEAD_DIM
                          + t * p.kv_heads * HEAD_DIM
                          + kv_head * HEAD_DIM;

            float score = 0.0f;
            for (uint i = tid; i < HEAD_DIM; i += 32) {
                score += float(q_exact[head * HEAD_DIM + i])
                       * float(hot_pool[k_offset + i]);
            }

            // SIMD reduction across 32 threads to get full dot product
            score = simd_reduce_add(score, tid);

            // Value pointer for this token
            const device half* v_row = &hot_pool[k_offset];

            // Online softmax update
            online_softmax_update(score, m, l, acc, v_row, head_dim_chunk, tid);
        }
    }

    // ─── Pass 2: Cold segment (decompression-free O(4) attention) ────

    for (uint b = 0; b < p.cold_count; b++) {
        uint block_ref = loom_refs[p.hot_count + b];
        uint phys_block = block_ref & 0x7FFFFFFF;

        for (uint t = 0; t < BLOCK_SIZE; t++) {
            // SparseCode for this token in this KV head
            uint code_idx = phys_block * BLOCK_SIZE * p.kv_heads
                          + t * p.kv_heads
                          + kv_head;

            SparseCode code = cold_pool[code_idx];

            // Decompression-free score: (Q · D) · C
            // q_latent is pre-computed as Q projected into dictionary space
            float score = 0.0f;
            for (uint i = 0; i < SPARSITY_K; i++) {
                score += float(q_latent[head * DICT_SIZE + code.indices[i]])
                       * float(code.coeffs[i]);
            }

            // Reconstruct value via dictionary: V ≈ D^T · C
            // V[dim] = Σ_k coeffs[k] * dict[kv_head][indices[k]][dim]
            //
            // Online softmax update with reconstructed value
            float new_m = max(m, score);
            float exp_old = exp(m - new_m);
            float exp_new = exp(score - new_m);

            for (uint i = 0; i < head_dim_chunk; i++) {
                acc[i] *= exp_old;
            }
            l = l * exp_old + exp_new;
            m = new_m;

            // Accumulate reconstructed value: V[dim] = Σ_k C_k · D[kv_head][idx_k][dim]
            for (uint i = 0; i < head_dim_chunk; i++) {
                uint dim_idx = tid + i * 32;
                if (dim_idx < HEAD_DIM) {
                    float v_reconstructed = 0.0f;
                    for (uint k = 0; k < SPARSITY_K; k++) {
                        // dict layout: [kv_heads, dict_size, head_dim]
                        uint dict_idx = kv_head * DICT_SIZE * HEAD_DIM
                                      + uint(code.indices[k]) * HEAD_DIM
                                      + dim_idx;
                        v_reconstructed += float(code.coeffs[k]) * float(dictionary[dict_idx]);
                    }
                    acc[i] += exp_new * v_reconstructed;
                }
            }
        }
    }

    // ─── Finalize: normalize and write output ────────────────────────

    // Normalize by softmax denominator
    if (l > 0.0f) {
        for (uint i = 0; i < head_dim_chunk; i++) {
            acc[i] /= l;
        }
    }

    // Write output token: each thread writes its HEAD_DIM/32 elements
    for (uint i = 0; i < head_dim_chunk; i++) {
        uint dim_idx = tid + i * 32;
        if (dim_idx < HEAD_DIM) {
            output[head * HEAD_DIM + dim_idx] = half(acc[i]);
        }
    }
}
