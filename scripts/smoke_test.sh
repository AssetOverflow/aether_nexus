#!/bin/bash
# AetherNexus Smoke Test Suite
# Tests various prompt difficulties and tool usage.

set -e

echo "╔══════════════════════════════════════════════════╗"
echo "║          AetherNexus Smoke Test Suite            ║"
echo "╚══════════════════════════════════════════════════╝"

# Ensure binary is built
cargo build --release -p nexus-core

BINARY="./target/release/nexus-core"
AETHER_FILE="smoke_test.aether"

# 1. Level: SIMPLE (Greeting & Identity)
echo -e "\n[TEST 1] Difficulty: SIMPLE"
echo "Prompt: 'Who are you and what is your purpose?'"
$BINARY --generate "Who are you and what is your purpose?" $AETHER_FILE --model models/inference/qwen2.5-0.5b-instruct | grep -t "assistant" || true

# 2. Level: INTERMEDIATE (Tool Usage - File System)
echo -e "\n[TEST 2] Difficulty: INTERMEDIATE"
echo "Prompt: 'List the files in the current directory and read the README.md if it exists.'"
$BINARY --generate "List the files in the current directory and read the README.md if it exists." $AETHER_FILE --model models/inference/qwen2.5-0.5b-instruct | grep -A 5 "Executing:" || true

# 3. Level: DIFFICULT (Reasoning & Code)
echo -e "\n[TEST 3] Difficulty: DIFFICULT"
echo "Prompt: 'Analyze the safety of the SandboxPolicy in src/sandbox.rs and propose one improvement.'"
$BINARY --generate "Analyze the safety of the SandboxPolicy in src/sandbox.rs and propose one improvement." $AETHER_FILE --model models/inference/qwen2.5-0.5b-instruct | grep -A 10 "REASONING" || true

echo -e "\n[FINISH] Smoke tests completed."
rm -f $AETHER_FILE
