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
use nexus_core::bench::{BenchConfig, run_benchmark};
use nexus_core::cortex::Cortex;
use nexus_core::distiller::Distiller;
use nexus_core::fabric::{Fabric, create_genesis};
use nexus_core::inference::{InferenceConfig, InferenceEngine};
use nexus_core::ops::OpsEngine;
use nexus_core::tokenizer::Tokenizer;
use nexus_core::types::{FabricLayout, NexusConfig, DeepSeekR1_1_5B, Qwen05B};
use nexus_core::weaver::WeaverEngine;
use nexus_core::weight_loader::{load_weights, load_weights_from_fabric, serialize_weights};
use std::env;
use std::path::Path;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let boot_start = std::time::Instant::now();
    let args: Vec<String> = env::args().collect();
    let run_bench = args.iter().any(|a| a == "--bench");
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let show_thinking = args.iter().any(|a| a == "--show-thinking");

    // Initialize Logger
    nexus_core::logging::init(verbose, show_thinking);

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

    let draft_model_dir = {
        let mut found = None;
        if let Some(idx) = args.iter().position(|x| x == "-d" || x == "--draft") {
            found = args.get(idx + 1).cloned();
        }
        found
    };
    let aether_path = {
        // Skip args[0] (binary path) and args that are flags or flag values
        let flag_value_indices: std::collections::HashSet<usize> = args
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                if (a == "--generate" || a == "--model" || a == "-m"
                    || a == "--draft" || a == "-d" 
                    || a == "--prompt" || a == "-p") && i + 1 < args.len() {
                    Some(i + 1)
                } else {
                    None
                }
            })
            .collect();
        args.iter()
            .enumerate()
            .skip(1) // skip binary path
            .find(|(i, a)| !a.starts_with("-") && !flag_value_indices.contains(i))
            .map(|(_, path)| path.clone())
            .unwrap_or_else(|| "brain.aether".to_string())
    };

    if verbose {
        println!("╔══════════════════════════════════════════════════╗");
        println!("║   AetherNexus v1.3 – Sovereign Tensor Organism  ║");
        println!("║   Forging the Fabric on Apple Silicon...         ║");
        println!("╚══════════════════════════════════════════════════╝");
        println!();
    }

    // 0. Load Configuration
    let config_path = Path::new("nexus.toml");
    let nexus_config = if config_path.exists() {
        nexus_core::nexus_info!("Loading configuration from nexus.toml...");
        let config_str = std::fs::read_to_string(config_path).expect("Failed to read nexus.toml");
        toml::from_str(&config_str).expect("Failed to parse nexus.toml")
    } else {
        nexus_core::nexus_debug!("No nexus.toml found. Using default configurations.");
        NexusConfig::default()
    };

    nexus_core::nexus_debug!(
        "Agent Config: Max Reflection: {}, Max Tokens: {}",
        nexus_config.agent.max_reflection_steps, nexus_config.agent.max_tokens
    );
    nexus_core::nexus_debug!(
        "Memory Config: Distill Threshold: {}, REM Interval: {}s",
        nexus_config.memory.distill_entropy_threshold, nexus_config.memory.rem_interval_secs
    );

    // 1. Resolve .aether file path and create genesis if needed
    if !std::path::Path::new(&aether_path).exists() {
        nexus_core::nexus_info!("No .aether file found at '{}'. Creating genesis...", aether_path);

        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|_| "Ed25519 key generation failed")?;
        let key_pair = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
            .map_err(|_| "Ed25519 key parse failed")?;

        create_genesis::<DeepSeekR1_1_5B>(&aether_path, &key_pair)?;

        nexus_core::nexus_info!("Created signed .aether file at '{}'", aether_path);
        nexus_core::nexus_debug!(
            "Ed25519 public key: {:02x?}",
            &ring::signature::KeyPair::public_key(&key_pair).as_ref()[..8]
        );
    }

    nexus_core::nexus_debug!("Claiming Unified Memory from '{}'...", aether_path);
    let mut fabric = Fabric::<DeepSeekR1_1_5B>::boot(&aether_path)?;

    let total_mb = fabric.total_size() / (1024 * 1024);
    nexus_core::nexus_debug!("Fabric mapped: {} MB", total_mb);
    nexus_core::nexus_debug!(
        "Hot pool:  {} MB",
        fabric.regions.hot_pool_size / (1024 * 1024)
    );
    nexus_core::nexus_debug!(
        "Cold pool: {} MB",
        fabric.regions.cold_pool_size / (1024 * 1024)
    );
    nexus_core::nexus_debug!("Dictionary: {} KB", fabric.regions.dict_size / 1024);
    nexus_core::nexus_debug!(
        "Observation buffers: {} MB",
        fabric.regions.obs_size / (1024 * 1024)
    );
    nexus_core::nexus_debug!(
        "Holographic trace: {} MB",
        fabric.regions.trace_size / (1024 * 1024)
    );

    nexus_core::nexus_debug!("Initializing Unified Capability Cortex...");
    let sandbox_policy = if config_path.exists() {
        nexus_core::sandbox::SandboxPolicy::from_config(&nexus_config.security)
    } else {
        nexus_core::nexus_debug!("Using permissive DEV default sandbox policy for the agent.");
        nexus_core::sandbox::SandboxPolicy::dev_default()
    };
    nexus_core::nexus_debug!("Sandbox policy: {:?}", sandbox_policy);
    let mut cortex = nexus_core::cortex::Cortex::boot(sandbox_policy);
    nexus_core::nexus_debug!(
        "{} capabilities registered",
        cortex.capability_count()
    );
    nexus_core::nexus_debug!("{:?}", cortex);

    let metallib_path = option_env!("WEAVER_METALLIB");
    let mut weaver = if let Some(path) = metallib_path {
        nexus_core::nexus_debug!("Loading Metal kernel from '{}'...", path);
        match WeaverEngine::new(path) {
            Ok(engine) => {
                nexus_core::nexus_debug!("Device: {}", engine.device_name());
                nexus_core::nexus_debug!(
                    "Max threads/threadgroup: {}",
                    engine.max_threads_per_threadgroup()
                );
                Some(engine)
            }
            Err(e) => {
                nexus_core::nexus_warn!("Failed to create pipeline: {}", e);
                nexus_core::nexus_info!("GPU decode unavailable, organism runs in CPU-only mode.");
                None
            }
        }
    } else {
        nexus_core::nexus_debug!("No metallib path — GPU decode not available.");
        None
    };

    nexus_core::nexus_debug!("Persona pathways:");
    for (i, loom) in fabric.looms.iter().enumerate() {
        nexus_core::nexus_debug!(
            "  [{i}] {:?} – hot: {}, cold: {}, token_pos: {}",
            loom.persona, loom.hot_count, loom.cold_count, loom.token_pos
        );
    }

    // 6. Run benchmark if requested
    if run_bench {
        if let Some(ref mut engine) = weaver {
            println!("╔══════════════════════════════════════════════════╗");
            println!("║           GPU Benchmark Mode                     ║");
            println!("╚══════════════════════════════════════════════════╝");
            println!();

            let config = BenchConfig::default();
            let results =
                run_benchmark(engine, &config).map_err(|e| format!("Benchmark failed: {}", e))?;

            println!("{}", results);
        } else {
            println!("⚠️  Cannot run benchmark: no Weaver engine available.");
            println!("    Ensure Metal SDK is installed and metallib is compiled.");
        }
    }

    // 7. Generate text if requested
    if let Some(ref prompt) = generate_prompt {
        let ops_metallib = option_env!("OPS_METALLIB");
        if let Some(ops_path) = ops_metallib {
            nexus_core::nexus_debug!("Loading ops metallib from '{}'...", ops_path);
            let ops =
                OpsEngine::new(ops_path).map_err(|e| format!("Ops engine init failed: {}", e))?;

            nexus_core::nexus_debug!("Loading tokenizer...");
            let tokenizer = Tokenizer::from_dir(&model_dir)
                .map_err(|e| format!("Tokenizer load failed: {}", e))?;
            nexus_core::nexus_debug!("Vocab size: {}", tokenizer.vocab_size());

            nexus_core::nexus_debug!("Auto-detecting model from '{}'...", model_dir);
            let mut config = InferenceConfig::detect_from_dir(&model_dir)
                .map_err(|e| format!("Model detection failed: {}", e))?;
            config.max_tokens = nexus_config.agent.max_tokens; // Override from nexus.toml

            // ─── Unified Weight Loading ─────────────────────────────────────
            // Priority: Fabric (brain.aether) → safetensors (models/**) → error
            // If loading from safetensors, auto-embed into Fabric for next boot.
            let weights = if fabric.weights_embedded() {
                nexus_core::nexus_info!("Loading weights from Fabric...");
                let (num_layers, has_biases) = fabric
                    .weight_manifest()
                    .ok_or("Fabric has WGHT magic but corrupt manifest")?;
                load_weights_from_fabric(
                    fabric.weight_data(),
                    num_layers,
                    has_biases,
                    config.hidden_size,
                    config.q_heads,
                    config.kv_heads,
                    config.head_dim,
                    config.intermediate_size,
                    config.vocab_size,
                )
                .map_err(|e| format!("Fabric weight load failed: {}", e))?
            } else {
                nexus_core::nexus_info!("Loading weights from safetensors (first boot)...");
                let w = load_weights(&model_dir, config.num_layers)
                    .map_err(|e| format!("Weight loading failed: {}", e))?;

                // Auto-embed into Fabric for next boot
                nexus_core::nexus_debug!("Auto-embedding weights into brain.aether...");
                let (serialized, has_biases) = serialize_weights(&w);
                if let Err(e) = fabric.embed_weights(&serialized, config.num_layers, has_biases) {
                    nexus_core::nexus_warn!("Failed to embed weights into Fabric: {}", e);
                } else {
                    fabric.force_checkpoint().unwrap_or_else(|e| {
                        nexus_core::nexus_warn!("Checkpoint after embed failed: {}", e)
                    });
                    nexus_core::nexus_info!("Weights embedded into Fabric.");
                }

                w
            };

            nexus_core::nexus_debug!("Initializing inference engine...");
            let mut engine = InferenceEngine::new(ops.clone(), &weights, config.clone());

            let mut draft_engine = match draft_model_dir {
                Some(ref draft_dir) => {
                    nexus_core::nexus_info!("Initializing DRAFT engine from '{}'...", draft_dir);
                    let draft_config = InferenceConfig::detect_from_dir(draft_dir)
                        .map_err(|e| format!("Draft model detection failed: {}", e))?;
                    
                    // Initialize Fabric for draft model
                    let draft_aether_path = "draft.aether";
                    let mut draft_fabric = if std::path::Path::new(draft_aether_path).exists() {
                        nexus_core::nexus_info!("Loading DRAFT Fabric from '{}'...", draft_aether_path);
                        Fabric::<Qwen05B>::boot(draft_aether_path)
                            .map_err(|e| format!("Failed to load draft Fabric: {}", e))?
                    } else {
                        nexus_core::nexus_info!("No .aether found at '{}'. Creating DRAFT genesis...", draft_aether_path);
                        
                        let rng = ring::rand::SystemRandom::new();
                        let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng)
                            .map_err(|_| "Draft Ed25519 key generation failed")?;
                        let draft_key_pair = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
                            .map_err(|_| "Draft Ed25519 key parse failed")?;

                        nexus_core::fabric::create_genesis::<Qwen05B>(draft_aether_path, &draft_key_pair)
                            .map_err(|e| format!("Failed to create draft Fabric genesis: {}", e))?;
                            
                        Fabric::<Qwen05B>::boot_with_mode(draft_aether_path, nexus_core::fabric::BootMode::Dev)
                            .map_err(|e| format!("Failed to create draft Fabric: {}", e))?
                    };

                    let draft_weights = if draft_fabric.weights_embedded() {
                        nexus_core::nexus_info!("Loading DRAFT weights from Fabric...");
                        let (num_layers, has_biases) = draft_fabric
                            .weight_manifest()
                            .ok_or("Draft Fabric has WGHT magic but corrupt manifest")?;
                        
                        load_weights_from_fabric(
                            draft_fabric.weight_data(),
                            num_layers,
                            has_biases,
                            draft_config.hidden_size,
                            draft_config.q_heads,
                            draft_config.kv_heads,
                            draft_config.head_dim,
                            draft_config.intermediate_size,
                            draft_config.vocab_size,
                        )
                        .map_err(|e| format!("Draft Fabric weight load failed: {}", e))?
                    } else {
                        nexus_core::nexus_info!("Loading DRAFT weights from safetensors (first boot)...");
                        let w = load_weights(draft_dir, draft_config.num_layers)
                            .map_err(|e| format!("Draft weight loading failed: {}", e))?;
                            
                        // Auto-embed into Fabric for next boot
                        nexus_core::nexus_debug!("Auto-embedding DRAFT weights into draft.aether...");
                        let (serialized, has_biases) = serialize_weights(&w);
                        if let Err(e) = draft_fabric.embed_weights(&serialized, draft_config.num_layers, has_biases) {
                            nexus_core::nexus_warn!("Failed to embed DRAFT weights into Fabric: {}", e);
                        } else {
                            draft_fabric.force_checkpoint().unwrap_or_else(|e| {
                                nexus_core::nexus_warn!("Checkpoint after draft embed failed: {}", e)
                            });
                            nexus_core::nexus_info!("DRAFT Weights embedded into Fabric.");
                        }
                        w
                    };

                    Some(InferenceEngine::new(ops.clone(), &draft_weights, draft_config))
                }
                None => None,
            };

            let boot_elapsed = boot_start.elapsed();
            println!(
                "AetherNexus v1.3 — {} — ready ({:.1}s)",
                config.model_name,
                boot_elapsed.as_secs_f32()
            );
            println!();

            nexus_core::nexus_info!("[INFERENCE] Generating from prompt: \"{}\"", prompt);
            nexus_core::nexus_info!("─── Agent Cognitive Loop ─────────────────────────────");

            let persona = "Genesis";

            // Spawn HITL interrupt thread
            let (tx, rx) = std::sync::mpsc::channel::<String>();
            let fabric_arc = Arc::new(tokio::sync::Mutex::new(fabric));
            let distill_memory_config = nexus_config.memory.clone();

            let mut distiller = Distiller::<DeepSeekR1_1_5B>::new(distill_memory_config);
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

            let mut agent = AgentLoop::new(
                &mut engine, // target_engine
                draft_engine.as_mut(), // speculative decoding engine
                &tokenizer,
                &mut cortex,
                rx,
                nexus_config.agent.max_reflection_steps,
            );
            let output = agent.run(persona, prompt).map_err(|e| e.to_string())?;

            nexus_core::nexus_info!("");
            nexus_core::nexus_info!("──────────────────────────────────────────────────");
            nexus_core::nexus_info!("");

            nexus_core::nexus_info!("[TRACE] Archiving successful trajectory into Holographic Trace...");
            {
                let mut fab = fabric_arc.lock().await;
                fab.append_trace(&output);
                fab.checkpoint()
                    .map_err(|e| format!("Checkpoint flush failed: {:?}", e))?;
            }
        } else {
            nexus_core::nexus_error!("OPS_METALLIB not set. Rebuild with: cargo build -p nexus-core");
        }
    }

    // 8. Organism is alive
    if verbose {
        println!("╔══════════════════════════════════════════════════╗");
        println!("║     Fabric claimed. Organism alive.              ║");
        println!("║     The M1 has its mind.                         ║");
        println!("╚══════════════════════════════════════════════════╝");
        println!();
        println!("The cognitive loop would now enter the eternal cycle:");
        println!("  1. Weaver decode (GPU) → action tensor");
        println!("  2. Cortex dispatch → zero-copy mutation");
        println!("  3. Checkpoint persistence → holographic trace");
        println!("  4. Next persona Loom activates");
        println!("  5. Background ANE distillation");
        println!();
        println!("Forge eternal.");
    }

    Ok(())
}
