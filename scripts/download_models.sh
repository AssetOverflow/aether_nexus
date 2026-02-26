#!/bin/bash
# Throttle-proof model downloader for AetherNexus
# Uses curl with resume (-C -) inside an infinite retry loop.
# Safe to run multiple times — it picks up where it left off.
# Kill with Ctrl-C at any time, then re-run to continue.

set -e

BASE_DIR="/Users/kaizenpro/Desktop/aether_nexus/models/inference"
QWEN_DIR="$BASE_DIR/qwen2.5-0.5b-instruct"
DEEPSEEK_DIR="$BASE_DIR/deepseek-r1-distill-qwen-1.5b"

mkdir -p "$QWEN_DIR"
mkdir -p "$DEEPSEEK_DIR"

# Retry-loop download: keeps retrying until curl exits 0 (complete)
download_until_done() {
    local url=$1
    local dest=$2
    local attempt=0

    while true; do
        attempt=$((attempt + 1))
        echo "[ATTEMPT $attempt] Downloading $(basename "$dest")..."
        
        # curl -C - resumes from where it left off
        # Exit code 0 = complete, 33 = range error (already complete)
        if curl -L -C - --retry 5 --retry-delay 2 --connect-timeout 15 \
                --max-time 600 --speed-limit 1000 --speed-time 30 \
                "$url" -o "$dest"; then
            echo "[DONE] $(basename "$dest") downloaded successfully!"
            return 0
        fi
        
        local exit_code=$?
        if [ $exit_code -eq 33 ]; then
            echo "[DONE] $(basename "$dest") already fully downloaded."
            return 0
        fi
        
        echo "[RETRY] Download interrupted (exit $exit_code). Waiting 5s before retry..."
        sleep 5
    done
}

echo "╔══════════════════════════════════════════════════╗"
echo "║  AetherNexus Model Downloader (Throttle-Proof)  ║"
echo "║  Safe to Ctrl-C and re-run at any time.         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── System 1: Qwen 2.5 0.5B Instruct (~1.0 GB) ──
echo "━━━ System 1: Qwen 2.5 0.5B Instruct ━━━"
QWEN_REPO="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main"

download_until_done "$QWEN_REPO/config.json?download=true"            "$QWEN_DIR/config.json"
download_until_done "$QWEN_REPO/generation_config.json?download=true"  "$QWEN_DIR/generation_config.json"
download_until_done "$QWEN_REPO/tokenizer.json?download=true"         "$QWEN_DIR/tokenizer.json"
download_until_done "$QWEN_REPO/tokenizer_config.json?download=true"  "$QWEN_DIR/tokenizer_config.json"
download_until_done "$QWEN_REPO/vocab.json?download=true"             "$QWEN_DIR/vocab.json"
download_until_done "$QWEN_REPO/merges.txt?download=true"             "$QWEN_DIR/merges.txt"
# The big one (~1GB) — this is the one that might need multiple retries
download_until_done "$QWEN_REPO/model.safetensors?download=true"      "$QWEN_DIR/model.safetensors"

echo ""
echo "✅ System 1 (Qwen 0.5B) complete!"
echo ""

# ── System 2: DeepSeek R1 Distill Qwen 1.5B (~3.0 GB) ──
echo "━━━ System 2: DeepSeek-R1-Distill-Qwen-1.5B ━━━"
DS_REPO="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main"

download_until_done "$DS_REPO/config.json?download=true"            "$DEEPSEEK_DIR/config.json"
download_until_done "$DS_REPO/generation_config.json?download=true"  "$DEEPSEEK_DIR/generation_config.json"
download_until_done "$DS_REPO/tokenizer.json?download=true"         "$DEEPSEEK_DIR/tokenizer.json"
download_until_done "$DS_REPO/tokenizer_config.json?download=true"  "$DEEPSEEK_DIR/tokenizer_config.json"
# The big one (~3GB) — this will definitely need multiple retries
download_until_done "$DS_REPO/model.safetensors?download=true"      "$DEEPSEEK_DIR/model.safetensors"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅ ALL MODELS DOWNLOADED SUCCESSFULLY           ║"
echo "║  System 1: $QWEN_DIR"
echo "║  System 2: $DEEPSEEK_DIR"
echo "╚══════════════════════════════════════════════════╝"
