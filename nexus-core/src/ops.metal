//  ops.metal – Transformer operation kernels for AetherNexus
//
//  These kernels implement the operations needed for the full Granite/Llama
//  transformer forward pass on Apple Silicon GPU.
//
//  All tensors are f16 for maximum SIMD throughput on M1.

#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// 0. GPU Buffer Copy — replaces CPU memcpy to keep everything in one cmd buffer
//    src_offset/dst_offset are in ELEMENTS (half), not bytes.
// ─────────────────────────────────────────────────────────────────────────────

kernel void copy_buffer(
    const device half* src       [[buffer(0)]],
    device half*       dst       [[buffer(1)]],
    constant uint&     count     [[buffer(2)]],
    constant uint&     src_off   [[buffer(3)]],
    constant uint&     dst_off   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    dst[dst_off + gid] = src[src_off + gid];
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Embedding Lookup
//    output[i] = embed_table[token_ids[i]]
// ─────────────────────────────────────────────────────────────────────────────

kernel void embed_lookup(
    const device uint*  token_ids    [[buffer(0)]],
    const device half*  embed_table  [[buffer(1)]],
    device half*        output       [[buffer(2)]],
    constant uint&      hidden_size  [[buffer(3)]],
    constant uint&      seq_len      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint token_idx = gid.y;
    uint dim_idx   = gid.x;

    if (token_idx >= seq_len || dim_idx >= hidden_size) return;

    uint token_id = token_ids[token_idx];
    output[token_idx * hidden_size + dim_idx] =
        embed_table[token_id * hidden_size + dim_idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. RMS Norm
//    output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]
//
//    Each threadgroup processes one token (one row of hidden_size).
//    Uses shared memory reduction for the sum of squares.
// ─────────────────────────────────────────────────────────────────────────────

kernel void rms_norm(
    const device half*  input        [[buffer(0)]],
    const device half*  weight       [[buffer(1)]],
    device half*        output       [[buffer(2)]],
    constant uint&      hidden_size  [[buffer(3)]],
    constant float&     eps          [[buffer(4)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    // Each threadgroup handles one token
    uint token_offset = gid * hidden_size;

    // Phase 1: compute partial sum of squares
    threadgroup float shared_sum[256];
    float partial_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tgs) {
        float val = float(input[token_offset + i]);
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: reduce
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: normalize
    float rms_inv = rsqrt(shared_sum[0] / float(hidden_size) + eps);

    for (uint i = tid; i < hidden_size; i += tgs) {
        float val = float(input[token_offset + i]);
        float w = float(weight[i]);
        output[token_offset + i] = half(val * rms_inv * w);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Rotary Position Embedding (RoPE)
//    Applies rotary embeddings to Q and K tensors.
//    Uses the standard cos/sin rotation on pairs of dimensions.
// ─────────────────────────────────────────────────────────────────────────────

kernel void rope(
    device half*        q            [[buffer(0)]],
    device half*        k            [[buffer(1)]],
    constant uint&      seq_len      [[buffer(2)]],
    constant uint&      q_heads      [[buffer(3)]],
    constant uint&      kv_heads     [[buffer(4)]],
    constant uint&      head_dim     [[buffer(5)]],
    constant uint&      position     [[buffer(6)]],
    constant float&     theta_base   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    // gid.x = pair index within head (0..head_dim/2)
    // gid.y = head index
    uint pair = gid.x;
    uint head = gid.y;
    uint half_dim = head_dim / 2;

    if (pair >= half_dim) return;

    // Compute the rotation angle
    float freq = 1.0f / pow(theta_base, float(2 * pair) / float(head_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Apply to Q heads
    if (head < q_heads) {
        uint base = head * head_dim;
        float q0 = float(q[base + pair]);
        float q1 = float(q[base + pair + half_dim]);
        q[base + pair]            = half(q0 * cos_val - q1 * sin_val);
        q[base + pair + half_dim] = half(q0 * sin_val + q1 * cos_val);
    }

    // Apply to K heads (fewer than Q due to GQA)
    if (head < kv_heads) {
        uint base = head * head_dim;
        float k0 = float(k[base + pair]);
        float k1 = float(k[base + pair + half_dim]);
        k[base + pair]            = half(k0 * cos_val - k1 * sin_val);
        k[base + pair + half_dim] = half(k0 * sin_val + k1 * cos_val);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Matrix Multiply (f16) — Tiled with shared memory
//    C = A × B^T
//    A: [M, K], B: [N, K] (row-major, B is transposed)
//    C: [M, N]
//
//    Each threadgroup computes a TILE×TILE block of C using shared memory.
//    Reduces global memory traffic by a factor of TILE_SIZE (~16×).
//    Dispatch: threadgroups=(ceil(N/16), ceil(M/16)), threads=(16,16).
// ─────────────────────────────────────────────────────────────────────────────

constant uint TILE_SIZE = 16;

kernel void matmul_f16(
    const device half*  A            [[buffer(0)]],
    const device half*  B            [[buffer(1)]],
    device half*        C            [[buffer(2)]],
    constant uint&      M            [[buffer(3)]],
    constant uint&      N            [[buffer(4)]],
    constant uint&      K            [[buffer(5)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 bid  [[threadgroup_position_in_grid]])
{
    uint tx = tid.x;  // column within tile (0..15)
    uint ty = tid.y;  // row within tile (0..15)

    uint row = bid.y * TILE_SIZE + ty;
    uint col = bid.x * TILE_SIZE + tx;

    threadgroup half tileA[TILE_SIZE][TILE_SIZE];
    threadgroup half tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        // Load A tile: A[row, t*TILE + tx]
        uint a_k = t * TILE_SIZE + tx;
        tileA[ty][tx] = (row < M && a_k < K) ? A[row * K + a_k] : half(0);

        // Load B tile: B[col_base + ty, t*TILE + tx]
        // col_base = bid.x * TILE_SIZE — the output columns this group handles
        uint b_row = bid.x * TILE_SIZE + ty;
        uint b_k = t * TILE_SIZE + tx;
        tileB[ty][tx] = (b_row < N && b_k < K) ? B[b_row * K + b_k] : half(0);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: C[row, col] += Σ_k A[row, t*T+k] * B[col, t*T+k]
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(tileA[ty][k]) * float(tileB[tx][k]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = half(sum);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4b. Vector-Matrix Multiply (f16) — SIMD-optimized for M=1 (decode path)
//     C[1,N] = A[1,K] × B[N,K]^T
//
//     Each SIMD group (32 threads) computes one output column's dot product.
//     32 threads read 32 contiguous elements → perfectly coalesced memory access.
//     Reduction via simd_shuffle_xor — zero shared memory, zero barriers.
//     8 output columns per threadgroup (256 threads / 32 per SIMD group).
//     Dispatch: threadgroups=(ceil(N/8),1,1), threads=(256,1,1).
// ─────────────────────────────────────────────────────────────────────────────

constant uint VECMAT_TG = 256;
constant uint VECMAT_COLS = 8;  // output columns per threadgroup (256 / 32)

kernel void vecmat_f16(
    const device half*  A            [[buffer(0)]],
    const device half*  B            [[buffer(1)]],
    device half*        C            [[buffer(2)]],
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint bid  [[threadgroup_position_in_grid]])
{
    uint simd_id = tid / 32;      // which SIMD group (0..7) = which output column
    uint lane    = tid % 32;       // lane within SIMD group

    uint col = bid * VECMAT_COLS + simd_id;
    if (col >= N) return;

    // Each lane handles every 32nd element of the dot product → coalesced reads
    float sum = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        sum += float(A[k]) * float(B[col * K + k]);
    }

    // SIMD reduction — no shared memory, no barriers
    sum += simd_shuffle_xor(sum, 16);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);

    if (lane == 0) {
        C[col] = half(sum);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4c. Vector-Matrix Multiply Scaled (f32 output) — for lm_head with M=1
//     C[1,N] = A[1,K] × B[N,K]^T * scale
//     Same SIMD strategy as vecmat_f16, outputs f32 for logit precision.
// ─────────────────────────────────────────────────────────────────────────────

kernel void vecmat_scaled(
    const device half*  A            [[buffer(0)]],
    const device half*  B            [[buffer(1)]],
    device float*       C            [[buffer(2)]],
    constant uint&      N            [[buffer(3)]],
    constant uint&      K            [[buffer(4)]],
    constant float&     scale        [[buffer(5)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint bid  [[threadgroup_position_in_grid]])
{
    uint simd_id = tid / 32;
    uint lane    = tid % 32;

    uint col = bid * VECMAT_COLS + simd_id;
    if (col >= N) return;

    float sum = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        sum += float(A[k]) * float(B[col * K + k]);
    }

    sum += simd_shuffle_xor(sum, 16);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);

    if (lane == 0) {
        C[col] = sum * scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. SiLU-gated activation (fused)
//    output = SiLU(gate) * up = (gate * sigmoid(gate)) * up
// ─────────────────────────────────────────────────────────────────────────────

kernel void silu_gate(
    const device half*  gate         [[buffer(0)]],
    const device half*  up           [[buffer(1)]],
    device half*        output       [[buffer(2)]],
    constant uint&      size         [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    float g = float(gate[gid]);
    float u = float(up[gid]);
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = half(silu_g * u);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Residual Add (in-place)
//    x += residual
// ─────────────────────────────────────────────────────────────────────────────

kernel void add_residual(
    device half*        x            [[buffer(0)]],
    const device half*  residual     [[buffer(1)]],
    constant uint&      size         [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    x[gid] = half(float(x[gid]) + float(residual[gid]));
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Scaled matmul for logits — Tiled with shared memory
//    Used for the final lm_head projection.
//    C = A × B^T * scale  (output in f32 for logit precision)
//    Same tiling strategy as matmul_f16.
// ─────────────────────────────────────────────────────────────────────────────

kernel void matmul_scaled(
    const device half*  A            [[buffer(0)]],
    const device half*  B            [[buffer(1)]],
    device float*       C            [[buffer(2)]],
    constant uint&      M            [[buffer(3)]],
    constant uint&      N            [[buffer(4)]],
    constant uint&      K            [[buffer(5)]],
    constant float&     scale        [[buffer(6)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 bid  [[threadgroup_position_in_grid]])
{
    uint tx = tid.x;
    uint ty = tid.y;
    uint row = bid.y * TILE_SIZE + ty;
    uint col = bid.x * TILE_SIZE + tx;

    threadgroup half tileA[TILE_SIZE][TILE_SIZE];
    threadgroup half tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_k = t * TILE_SIZE + tx;
        tileA[ty][tx] = (row < M && a_k < K) ? A[row * K + a_k] : half(0);

        uint b_row = bid.x * TILE_SIZE + ty;
        uint b_k = t * TILE_SIZE + tx;
        tileB[ty][tx] = (b_row < N && b_k < K) ? B[b_row * K + b_k] : half(0);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(tileA[ty][k]) * float(tileB[tx][k]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum * scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Causal Self-Attention (single-head, for direct forward pass)
//    Computes scaled dot-product attention:
//    attn = softmax(Q × K^T / sqrt(d_k)) × V
//
//    Q: [1, head_dim]  (current token query)
//    K: [seq_len, head_dim]  (cached keys)
//    V: [seq_len, head_dim]  (cached values)
//    output: [1, head_dim]
// ─────────────────────────────────────────────────────────────────────────────

kernel void causal_attention(
    const device half*  Q            [[buffer(0)]],
    const device half*  K            [[buffer(1)]],
    const device half*  V            [[buffer(2)]],
    device half*        output       [[buffer(3)]],
    constant uint&      kv_len       [[buffer(4)]],
    constant uint&      head_dim     [[buffer(5)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    // Phase 1: compute attention scores Q·K^T / sqrt(d)
    float scale = rsqrt(float(head_dim));

    // Use online softmax to avoid materializing the full score vector
    float m = -INFINITY;  // running max
    float l = 0.0f;       // running sum of exp

    // Accumulator for weighted V
    // Thread handles a chunk of head_dim
    uint dim_chunk = (head_dim + tgs - 1) / tgs;

    // We need threadgroup memory for the scores
    threadgroup float scores[4096]; // max kv_len

    // Phase 1: compute all scores
    for (uint pos = tid; pos < kv_len; pos += tgs) {
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += float(Q[d]) * float(K[pos * head_dim + d]);
        }
        scores[pos] = score * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: find max for numerical stability
    threadgroup float shared_max[256];
    float local_max = -INFINITY;
    for (uint pos = tid; pos < kv_len; pos += tgs) {
        local_max = max(local_max, scores[pos]);
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared_max[0];

    // Phase 3: compute exp and sum
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint pos = tid; pos < kv_len; pos += tgs) {
        scores[pos] = exp(scores[pos] - global_max);
        local_sum += scores[pos];
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_sum = shared_sum[0];

    // Phase 4: normalize scores
    for (uint pos = tid; pos < kv_len; pos += tgs) {
        scores[pos] /= total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 5: accumulate weighted V
    for (uint d = tid; d < head_dim; d += tgs) {
        float acc = 0.0f;
        for (uint pos = 0; pos < kv_len; pos++) {
            acc += scores[pos] * float(V[pos * head_dim + d]);
        }
        output[d] = half(acc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Multi-Head GQA Attention (all heads in one dispatch!)
//    Each threadgroup = one query head. GQA: kv_head = q_head / gqa_group
//    Q:       [q_heads * head_dim]
//    K/V_cache: [seq_len * kv_heads * head_dim]
//    output:  [q_heads * head_dim]
// ─────────────────────────────────────────────────────────────────────────────

kernel void multihead_attention(
    const device half*  Q            [[buffer(0)]],
    const device half*  K_cache      [[buffer(1)]],
    const device half*  V_cache      [[buffer(2)]],
    device half*        output       [[buffer(3)]],
    constant uint&      kv_len       [[buffer(4)]],
    constant uint&      head_dim     [[buffer(5)]],
    constant uint&      q_heads      [[buffer(6)]],
    constant uint&      kv_heads     [[buffer(7)]],
    uint                q_head       [[threadgroup_position_in_grid]],
    uint                tid          [[thread_index_in_threadgroup]],
    uint                tgs          [[threads_per_threadgroup]])
{
    if (q_head >= q_heads) return;
    
    uint gqa_group = q_heads / kv_heads;
    uint kv_head = q_head / gqa_group;
    uint kv_stride = kv_heads * head_dim;
    
    float scale = rsqrt(float(head_dim));
    
    const device half* Q_head = Q + q_head * head_dim;
    device half* out_head = output + q_head * head_dim;
    
    threadgroup float scores[4096];
    
    // Phase 1: Q·K^T / sqrt(d)
    for (uint pos = tid; pos < kv_len; pos += tgs) {
        float score = 0.0f;
        const device half* K_pos = K_cache + pos * kv_stride + kv_head * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            score += float(Q_head[d]) * float(K_pos[d]);
        }
        scores[pos] = score * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: max for stability
    threadgroup float shared_max[256];
    float local_max = -INFINITY;
    for (uint pos = tid; pos < kv_len; pos += tgs) {
        local_max = max(local_max, scores[pos]);
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared_max[0];
    
    // Phase 3: exp + sum
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint pos = tid; pos < kv_len; pos += tgs) {
        scores[pos] = exp(scores[pos] - global_max);
        local_sum += scores[pos];
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_sum = shared_sum[0];
    
    // Phase 4: normalize
    for (uint pos = tid; pos < kv_len; pos += tgs) {
        scores[pos] /= total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 5: weighted sum of V
    for (uint d = tid; d < head_dim; d += tgs) {
        float acc = 0.0f;
        for (uint pos = 0; pos < kv_len; pos++) {
            acc += scores[pos] * float(V_cache[pos * kv_stride + kv_head * head_dim + d]);
        }
        out_head[d] = half(acc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Scale f16 (in-place)
//     x[i] *= scale  (for embedding/residual/attention multipliers)
//
//     Eliminates CPU←GPU round-trips for scalar multiplication.
//     Granite uses ~3 scaling steps per layer; this kernel lets all
//     of them stay within a single GPU command buffer batch.
// ─────────────────────────────────────────────────────────────────────────────

kernel void scale_f16(
    device half*        x            [[buffer(0)]],
    constant uint&      count        [[buffer(1)]],
    constant float&     scale        [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    x[gid] = half(float(x[gid]) * scale);
}
