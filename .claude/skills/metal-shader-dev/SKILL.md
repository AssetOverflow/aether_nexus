---
name: Metal Shader Development
description: Activate this skill when writing, editing, or debugging Metal Shading Language (MSL) code in nexus-core/src/*.metal files, or when discussing Metal GPU dispatch from Rust (weaver.rs, ops.rs, build.rs). Also activate when the topic involves threadgroup sizing, SIMD groups, metal::half, buffer binding indices, or the build.rs shader compilation pipeline.
version: 1.0.0
---

# Metal Shader Development — AetherNexus

## Project Shader Files

| File | Purpose |
|------|---------|
| `nexus-core/src/weaver_kernel.metal` | Decompression-free sparse attention decode (hot + cold paths) |
| `nexus-core/src/ops.metal` | Transformer ops: matmul, RMSNorm, masked attention, FFN, SiLU |
| `nexus-core/build.rs` | Compilation: MSL → AIR → metallib via xcrun |

## Metal Compilation Pipeline

```bash
# MSL → AIR
xcrun -sdk macosx metal -c <src>.metal -o <out>.air -std=metal3.0 -O2

# AIR → metallib
xcrun -sdk macosx metallib <in>.air -o <out>.metallib
```

`build.rs` exports `WEAVER_METALLIB` and `OPS_METALLIB` env vars. The Rust side loads them at runtime via `metal::Library::new_with_source` or from the compiled `.metallib` bytes embedded via `include_bytes!`.

## M1/M2/M3 GPU Constraints

- **Max threads per threadgroup**: 1024
- **SIMD width**: 32 lanes
- **Threadgroup memory**: 32 KB (use sparingly; bank conflicts hurt)
- **Half precision**: `metal::half` / `half` — native f16 arithmetic on Apple GPU
- **Register pressure**: Prefer smaller threadgroups (e.g., 128 threads) if register count is high

## Buffer Binding Convention

Bindings must match exactly between MSL `[[buffer(N)]]` and Rust `encoder.set_buffer(N, ...)`.

```metal
// MSL
kernel void my_kernel(
    device const half* weights [[buffer(0)]],
    device half*       output  [[buffer(1)]],
    constant uint&     seq_len [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) { ... }
```

```rust
// Rust (ops.rs or weaver.rs)
encoder.set_buffer(0, Some(&weights_buf), 0);
encoder.set_buffer(1, Some(&output_buf), 0);
encoder.set_bytes(2, 4, &seq_len as *const u32 as *const _);
```

## SparseCode Cold Path

The cold attention path reconstructs KV vectors from `SparseCode` without decompression:

```metal
// SparseCode: 16 bytes total
// indices: 4 × u16  (dictionary lookup 0..511)
// coeffs:  4 × half (sparse weights)
struct SparseCode {
    ushort4 indices;
    half4   coeffs;
};

// Sparse dot product (O(4) ops per KV vector):
half sparse_dot(device const half* dict_row, SparseCode code) {
    return dot(half4(dict_row[code.indices.x],
                     dict_row[code.indices.y],
                     dict_row[code.indices.z],
                     dict_row[code.indices.w]),
               code.coeffs);
}
```

## GQA Head Broadcasting

The project uses Grouped Query Attention (GQA): 32 Q heads, 8 KV heads. KV head index = `q_head / 4`.

```metal
uint kv_head = q_head / 4u;  // 32Q / 8KV = 4 Q heads per KV head
```

## Softmax Stability Pattern

Always subtract the max before exp to prevent overflow with half precision:

```metal
half max_val = -HALF_MAX;
for (uint i = 0; i < seq_len; i++) max_val = max(max_val, scores[i]);
half sum = 0.0h;
for (uint i = 0; i < seq_len; i++) { scores[i] = exp(scores[i] - max_val); sum += scores[i]; }
for (uint i = 0; i < seq_len; i++) scores[i] /= sum;
```

## Threadgroup Barrier Rules

Always insert a barrier before reading threadgroup memory written by other threads:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
```

Insert between the write phase and read phase of any reduction or shared memory pattern.

## Common Build Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `xcrun: error: unable to find utility "metal"` | Xcode CLT not installed | `xcode-select --install` |
| `error: use of undeclared identifier 'half4'` | Wrong Metal standard | Add `-std=metal3.0` |
| `Shader validation error: buffer binding X` | Binding index mismatch | Match `[[buffer(N)]]` with Rust `set_buffer(N, ...)` |
| `metallib: error` | AIR file corrupt | Rebuild from MSL, check for compile warnings |
