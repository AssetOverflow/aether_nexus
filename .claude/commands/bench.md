---
description: Run the GPU benchmark and compare results against performance targets
allowed-tools: Bash
---

Run the GPU benchmark:

```
!`cargo run -p nexus-core --release -- --bench 2>&1`
```

After capturing the output, evaluate the results against the project's performance targets:

| Metric | Target |
|--------|--------|
| Sustained throughput | 92–118 tokens/sec |
| Agent cycle latency | 35–65 ms |
| Cold start | <800 ms |
| Power | 0.55–0.95 W |

For each metric:
- Show the measured value alongside the target.
- Mark as ✓ PASS, ⚠ MARGINAL (within 20% of target), or ✗ FAIL.

If any metric fails or is marginal:
1. Identify the most likely bottleneck (threadgroup occupancy, memory bandwidth, cold pool ratio, CPU wake-ups).
2. Suggest specific changes to `ops.metal`, `weaver_kernel.metal`, `weaver.rs`, or `distiller.rs` that could recover performance.
3. Note whether a Metal shader recompilation is needed (`cargo build --workspace`).
