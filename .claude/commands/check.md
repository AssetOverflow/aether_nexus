---
description: Run cargo check + clippy across the workspace and summarize all errors and warnings
allowed-tools: Bash
---

Run the following commands and capture all output:

```
!`cargo check --workspace --message-format=short 2>&1`
!`cargo clippy --workspace -- -D warnings 2>&1`
```

Then:
1. Summarize the total error count and warning count.
2. Group errors by file, showing file:line and the error message.
3. For each error, suggest the most likely fix based on the AetherNexus codebase patterns (Pod/Zeroable constraints, repr(C), Metal buffer bindings, etc.).
4. List warnings separately, flagging any that indicate real bugs (unused Result, dead_code on a capability handler, etc.).
5. If everything is clean, confirm with "✓ Workspace compiles cleanly."
