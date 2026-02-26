//! AetherNexus – The Organism Wakes
//!
//! This is the ignition sequence: the single entry point that boots the
//! Unified Fabric, initializes the Cortex, and enters the cognitive loop.
//!
//! # Usage
//!
//! ```bash
//! cargo run -- brain.aether                       # Normal boot
//! cargo run -- --bench                            # Run GPU benchmark
//! cargo run -- --generate "Hello, my name is"     # Generate text
//! ```

use nexus_core::agent::AgentLoop;
use nexus_core::bench::{run_benchmark, BenchConfig};
use nexus_core::cortex::Cortex;
use nexus_core::distiller::Distiller;
use nexus_core::fabric::{create_genesis, Fabric};
use nexus_core::inference::{InferenceConfig, InferenceEngine};
use nexus_core::ops::OpsEngine;
use nexus_core::tokenizer::Tokenizer;
use nexus_core::types::{FabricLayout, Granite2B, Llama8B, NexusConfig};
use nexus_core::weaver::WeaverEngine;
use nexus_core::weight_loader::load_weights;
use std::env;
use std::path::Path;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let run_bench = args.iter().any(|a| a == "--bench");
    let generate_prompt = {
        let mut found = None;
        for (i, arg) in args.iter().enumerate() {
            if arg == "--generate" {
                if i + 1 < args.len() {
                    found = Some(args[i + 1].clone());
                }
            }
        }
        found
    };
    let model_dir = {
        let mut found = None;
        if let Some(idx) = args.iter().position(|x| x == "-m" || x == "--model") {
            found = args.get(idx + 1).cloned();
        }
        found.unwrap_or_else(|| {
            // Try workspace root first, then nexus-core subdir
            if std::path::Path::new("models/inference/qwen2.5-0.5b-instruct").exists() {
                "models/inference/qwen2.5-0.5b-instruct".to_string()
            } else {
                "../models/inference/qwen2.5-0.5b-instruct".to_string()
            }
        })
    };
    let aether_path = {
        // Skip args[0] (binary path) and args that are flags or flag values
        let flag_value_indices: std::collections::HashSet<usize> = args.iter().enumerate()
            .filter_map(|(i, a)| {
                if (a == "--generate" || a == "--model" || a == "-m") && i + 1 < args.len() {
                    Some(i + 1)
                } else {
                    None
                }
            })
            .collect();
        args.iter().enumerate()
            .skip(1) // skip binary path
            .find(|(i, a)| !a.starts_with("-") && !flag_value_indices.contains(i))
            .map(|(_, path)| path.clone())
            .unwrap_or_else(|| "brain.aether".to_string())
    };

    println!("╔══════════════════════════════════════════════════╗");
    println!("║   AetherNexus v1.3 – Sovereign Tensor Organism  ║");
    println!("║   Forging the Fabric on Apple Silicon...         ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    // 0. Load Configuration
    let config_path = Path::new("nexus.toml");
    let nexus_config = if config_path.exists() {
        println!("[BOOT] Loading configuration from nexus.toml...");
        let config_str = std::fs::read_to_string(config_path)
            .expect("Failed to read nexus.toml");
        toml::from_str(&config_str).expect("Failed to parse nexus.toml")
    } else {
        println!("[BOOT] No nexus.toml found. Using default configurations.");
        NexusConfig::default()
    };

    println!("[BOOT] Agent Config: Max Reflection: {}, Max Tokens: {}", 
             nexus_config.agent.max_reflection_steps, nexus_config.agent.max_tokens);
    println!("[BOOT] Memory Config: Distill Threshold: {}, REM Interval: {}s",
             nexus_config.memory.distill_entropy_threshold, nexus_config.memory.rem_interval_secs);
    println!();

    // 1. Resolve .aether file path and create genesis if needed
    if !std::path::Path::new(&aether_path).exists() {
        println!("[GENESIS] No .aether file found at '{}'. Creating genesis...", aether_path);

        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|_| "Ed25519 key generation failed")?;
        let key_pair = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
            .map_err(|_| "Ed25519 key parse failed")?;

        create_genesis::<Llama8B>(&aether_path, &key_pair)?;

        println!("[GENESIS] Created signed .aether file at '{}'", aether_path);
        println!("[GENESIS] Ed25519 public key: {:02x?}", &ring::signature::KeyPair::public_key(&key_pair).as_ref()[..8]);
        println!();
    }

    // 2. Boot the Fabric – claim Unified Memory
    println!("[BOOT] Claiming Unified Memory from '{}'...", aether_path);
    let fabric = Fabric::<Llama8B>::boot(&aether_path)?;

    let total_mb = fabric.total_size() / (1024 * 1024);
    println!("[BOOT] Fabric mapped: {} MB", total_mb);
    println!("[BOOT] Hot pool:  {} MB", fabric.regions.hot_pool_size / (1024 * 1024));
    println!("[BOOT] Cold pool: {} MB", fabric.regions.cold_pool_size / (1024 * 1024));
    println!("[BOOT] Dictionary: {} KB", fabric.regions.dict_size / 1024);
    println!("[BOOT] Observation buffers: {} MB", fabric.regions.obs_size / (1024 * 1024));
    println!("[BOOT] Holographic trace: {} MB", fabric.regions.trace_size / (1024 * 1024));
    println!();

    // 3. Boot the Cortex
    println!("[CORTEX] Initializing Unified Capability Cortex...");
    let mut cortex = nexus_core::cortex::Cortex::boot();
    println!("[CORTEX] {} capabilities registered", cortex.capability_count());
    println!("[CORTEX] {:?}", cortex);
    println!();

    // 4. Boot the Weaver Engine (GPU pipeline)
    let metallib_path = option_env!("WEAVER_METALLIB");
    let weaver = if let Some(path) = metallib_path {
        println!("[WEAVER] Loading Metal kernel from '{}'...", path);
        match WeaverEngine::new(path) {
            Ok(engine) => {
                println!("[WEAVER] Device: {}", engine.device_name());
                println!("[WEAVER] Max threads/threadgroup: {}", engine.max_threads_per_threadgroup());
                println!();
                Some(engine)
            }
            Err(e) => {
                println!("[WEAVER] ⚠️  Failed to create pipeline: {}", e);
                println!("[WEAVER] GPU decode unavailable, organism runs in CPU-only mode.");
                println!();
                None
            }
        }
    } else {
        println!("[WEAVER] No metallib path — GPU decode not available.");
        println!();
        None
    };

    // 5. Report Loom state
    println!("[LOOMS] Persona pathways:");
    for (i, loom) in fabric.looms.iter().enumerate() {
        println!(
            "  [{i}] {:?} – hot: {}, cold: {}, token_pos: {}",
            loom.persona, loom.hot_count, loom.cold_count, loom.token_pos
        );
    }
    println!();

    // 6. Run benchmark if requested
    if run_bench {
        if let Some(ref engine) = weaver {
            println!("╔══════════════════════════════════════════════════╗");
            println!("║           GPU Benchmark Mode                     ║");
            println!("╚══════════════════════════════════════════════════╝");
            println!();

            let config = BenchConfig::default();
            let results = run_benchmark(engine, &config)
                .map_err(|e| format!("Benchmark failed: {}", e))?;

            println!("{}", results);
        } else {
            println!("⚠️  Cannot run benchmark: no Weaver engine available.");
            println!("    Ensure Metal SDK is installed and metallib is compiled.");
        }
    }

    // 7. Generate text if requested
    if let Some(ref prompt) = generate_prompt {
        println!("╔══════════════════════════════════════════════════╗");
        println!("║           Text Generation Mode                   ║");
        println!("╚══════════════════════════════════════════════════╝");
        println!();

        let ops_metallib = option_env!("OPS_METALLIB");
        if let Some(ops_path) = ops_metallib {
            println!("[OPS] Loading ops metallib from '{}'...", ops_path);
            let ops = OpsEngine::new(ops_path)
                .map_err(|e| format!("Ops engine init failed: {}", e))?;

            println!("[TOKENIZER] Loading tokenizer...");
            let tokenizer = Tokenizer::from_dir(&model_dir)
                .map_err(|e| format!("Tokenizer load failed: {}", e))?;
            println!("[TOKENIZER] Vocab size: {}", tokenizer.vocab_size());

            println!("[MODEL] Auto-detecting model from '{}'...", model_dir);
            let mut config = InferenceConfig::detect_from_dir(&model_dir)
                .map_err(|e| format!("Model detection failed: {}", e))?;
            config.max_tokens = nexus_config.agent.max_tokens; // Override from nexus.toml
            println!("[MODEL] Loaded: {}", config.model_name);
            println!("[MODEL] Architecture: {}h x {}L, {} heads, vocab {}, EOS={}",
                config.hidden_size, config.num_layers, config.q_heads, config.vocab_size, config.eos_token_id);
            
            let weights = load_weights(&model_dir, config.num_layers)
                .map_err(|e| format!("Weight loading failed: {}", e))?;

            println!("[INFERENCE] Initializing engine...");
            let mut engine = InferenceEngine::new(ops, &weights, config);

            println!("[INFERENCE] Generating from prompt: \"{}\"", prompt);
            println!();
            println!("─── Agent Cognitive Loop ─────────────────────────────");

            let persona = "Genesis";

            // Spawn HITL interrupt thread
            let (tx, rx) = std::sync::mpsc::channel::<String>();
            let fabric_arc = Arc::new(std::sync::Mutex::new(fabric));
            let distill_memory_config = nexus_config.memory.clone();
            
            let mut distiller = Distiller::<Llama8B>::new(distill_memory_config);
            let fabric_distiller_arc = fabric_arc.clone();
            
            // Spawns the background REM cycle thread
            tokio::spawn(async move {
                distiller.run(fabric_distiller_arc).await;
            });
            
            std::thread::spawn(move || {
                let stdin = std::io::stdin();
                for line in stdin.lines() {
                    if let Ok(text) = line {
                        if !text.trim().is_empty() {
                            let _ = tx.send(text.trim().to_string());
                        }
                    }
                }
            });

            let mut agent = AgentLoop::new(&mut engine, &tokenizer, &mut cortex, rx, nexus_config.agent.max_reflection_steps);
            let output = agent.run(persona, prompt).map_err(|e| e.to_string())?;

            println!();
            println!("──────────────────────────────────────────────────");
            println!();
            
            println!("[TRACE] Archiving successful trajectory into Holographic Trace...");
            if let Ok(mut fab) = fabric_arc.lock() {
                fab.append_trace(&output);
                fab.persist().map_err(|e| format!("WAL flush failed: {:?}", e))?;
            } else {
                eprintln!("Failed to acquire fabric lock to append trace.");
            }
        } else {
            println!("⚠️  OPS_METALLIB not set. Cannot run inference.");
            println!("    Rebuild with: cargo build -p nexus-core");
        }
    }

    // 8. Organism is alive
    println!("╔══════════════════════════════════════════════════╗");
    println!("║     Fabric claimed. Organism alive.              ║");
    println!("║     The M1 has its mind.                         ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("The cognitive loop would now enter the eternal cycle:");
    println!("  1. Weaver decode (GPU) → action tensor");
    println!("  2. Cortex dispatch → zero-copy mutation");
    println!("  3. WAL persistence → holographic trace");
    println!("  4. Next persona Loom activates");
    println!("  5. Background ANE distillation");
    println!();
    println!("Forge eternal.");

    Ok(())
}
