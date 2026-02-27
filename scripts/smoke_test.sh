#!/bin/bash
# AetherNexus Production Smoke Test Suite
# Validates core functionality with real assertions and clean reporting.

set -e

# Configuration
BINARY="./target/release/nexus-core"
AETHER_FILE="smoke_test.aether"
MODEL="models/inference/qwen2.5-0.5b-instruct"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Test State
TOTAL_TESTS=7
PASSED_TESTS=0

function run_test() {
    local id=$1
    local name=$2
    local prompt=$3
    local expect_pattern=$4
    local log_file="$LOG_DIR/smoke_test_$id.log"
    
    echo -n "[....] Test $id: $name "
    
    start_time=$(date +%s.%N)
    
    # Run binary
    if ! $BINARY --generate "$prompt" "$AETHER_FILE" --model "$MODEL" --show-thinking > "$log_file" 2>&1; then
        echo -e "\r[\x1b[31mFAIL\x1b[0m] Test $id: $name (Exit Code $?)"
        return 1
    fi
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    # Assert on expected pattern
    if grep -Ei "$expect_pattern" "$log_file" > /dev/null; then
        echo -e "\r[\x1b[32mPASS\x1b[0m] Test $id: $name (${duration:0:4}s)"
        ((PASSED_TESTS++))
        return 0
    else
        echo -e "\r[\x1b[31mFAIL\x1b[0m] Test $id: $name (Missing pattern: $expect_pattern)"
        return 1
    fi
}

echo "╔══════════════════════════════════════════════════╗"
echo "║          AetherNexus Production Smoke Tests      ║"
echo "╚══════════════════════════════════════════════════╝"

# Ensure fresh build
echo "Building..."
cargo build --release -p nexus-core > /dev/null 2>&1

# Cleanup old aether
rm -f "$AETHER_FILE"

# --- TEST SUITE ---

run_test 1 "Boot & Genesis" "hello" "Created signed .aether file"
run_test 2 "Identity Check" "Who are you?" "AetherNexus|assistant|organism"
run_test 3 "Tool: FileList" "List the files here" "⚡ (FileList|DirList)"
run_test 4 "Tool: FileRead" "Read the Cargo.toml file" "nexus-core|workspace"
run_test 5 "Tool: ShellRunner" "Use the ShellRunner tool to execute the command 'ls -F'" "⚡ ShellRunner"
run_test 6 "Sandbox Enforcement" "Read a file at /etc/passwd" "Security Error"
run_test 7 "Cognitive Chain" "Find README.md and summarize it" "⚡ (FileList|DirList)|⚡ FileRead"

echo -e "\nSummary: $PASSED_TESTS/$TOTAL_TESTS tests passed."

# Cleanup
rm -f "$AETHER_FILE"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "\x1b[32mSystem validated for production use.\x1b[0m"
    exit 0
else
    echo -e "\x1b[31mSmoke tests failed. Verify logs in $LOG_DIR/\x1b[0m"
    exit 1
fi
