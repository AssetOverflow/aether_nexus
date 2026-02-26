//! Agent Cognitive Loop (ReAct / Think / Execute)
//!
//! Ties the Cortex capabilities directly to the `InferenceEngine` streaming loop
//! to execute "Thought -> Action -> Observation" cycles autonomously.

use crate::inference::InferenceEngine;
use crate::tokenizer::Tokenizer;
use crate::cortex::Cortex;
use minijinja::{Environment, context};
use serde::Serialize;
use std::sync::mpsc::Receiver;

#[derive(Serialize)]
struct CapDesc {
    id: String,
    description: String,
}

pub struct AgentLoop<'a> {
    pub engine: &'a mut InferenceEngine,
    pub tokenizer: &'a Tokenizer,
    pub cortex: &'a mut Cortex,
    pub interrupt_rx: Receiver<String>,
    pub max_reflection_steps: usize,
}

impl<'a> AgentLoop<'a> {
    pub fn new(engine: &'a mut InferenceEngine, tokenizer: &'a Tokenizer, cortex: &'a mut Cortex, interrupt_rx: Receiver<String>, max_reflection_steps: usize) -> Self {
        Self { engine, tokenizer, cortex, interrupt_rx, max_reflection_steps }
    }

    /// Execute a text-based tool call and return the observation string.
    fn execute_tool(&self, tool_call: &str) -> String {
        // Parse "ToolName|arg" format
        let parts: Vec<&str> = tool_call.splitn(2, '|').collect();
        let tool_name = parts[0].trim();
        let tool_arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

        match tool_name {
            "ShellRunner" => {
                let policy = self.cortex.policy();
                let (binary, cmd_args) = match policy.validate_command(tool_arg) {
                    Ok(res) => res,
                    Err(e) => return format!("Security Error: {}", e),
                };
                let validated_cwd = match policy.validate_path(".") {
                    Ok(p) => p,
                    Err(e) => return format!("Security Error: {}", e),
                };
                let output = std::process::Command::new(binary)
                    .args(cmd_args)
                    .current_dir(validated_cwd)
                    .output();
                match output {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        let result = if !stdout.is_empty() { stdout.to_string() } else { stderr.to_string() };
                        // Truncate to avoid overwhelming the context
                        if result.len() > 500 {
                            format!("{}...[truncated]", &result[..500])
                        } else if result.is_empty() {
                            "(no output)".to_string()
                        } else {
                            result
                        }
                    }
                    Err(e) => format!("Error: {}", e),
                }
            }
            "FileRead" => {
                let policy = self.cortex.policy();
                let validated_path = match policy.validate_path(tool_arg) {
                    Ok(p) => p,
                    Err(e) => return format!("Security Error: {}", e),
                };
                match std::fs::read_to_string(&validated_path) {
                    Ok(content) => {
                        if content.len() > 500 {
                            format!("{}...[truncated]", &content[..500])
                        } else {
                            content
                        }
                    }
                    Err(e) => format!("Error reading '{}': {}", validated_path.display(), e),
                }
            }
            "DirList" => {
                let dir = if tool_arg.is_empty() { "." } else { tool_arg };
                let policy = self.cortex.policy();
                let validated_path = match policy.validate_path(dir) {
                    Ok(p) => p,
                    Err(e) => return format!("Security Error: {}", e),
                };
                match std::fs::read_dir(&validated_path) {
                    Ok(entries) => {
                        let mut listing = Vec::new();
                        for entry in entries.take(30) {
                            if let Ok(e) = entry {
                                let name = e.file_name().to_string_lossy().to_string();
                                let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                                listing.push(if is_dir { format!("{}/", name) } else { name });
                            }
                        }
                        listing.join("\n")
                    }
                    Err(e) => format!("Error listing '{}': {}", validated_path.display(), e),
                }
            }
            "GitStatus" => {
                let policy = self.cortex.policy();
                let validated_cwd = match policy.validate_path(".") {
                    Ok(p) => p,
                    Err(e) => return format!("Security Error: {}", e),
                };
                let output = std::process::Command::new("git")
                    .args(["status", "--short"])
                    .current_dir(validated_cwd)
                    .output();
                match output {
                    Ok(out) => {
                        let result = String::from_utf8_lossy(&out.stdout).to_string();
                        if result.is_empty() { "clean - no changes".to_string() } else { result }
                    }
                    Err(e) => format!("Error: {}", e),
                }
            }
            "CargoCheck" => {
                let policy = self.cortex.policy();
                let validated_cwd = match policy.validate_path(".") {
                    Ok(p) => p,
                    Err(e) => return format!("Security Error: {}", e),
                };
                let output = std::process::Command::new("cargo")
                    .args(["check", "--message-format", "short"])
                    .current_dir(validated_cwd)
                    .output();
                match output {
                    Ok(out) => {
                        let result = String::from_utf8_lossy(&out.stderr).to_string();
                        if result.len() > 500 {
                            format!("{}...[truncated]", &result[..500])
                        } else {
                            result
                        }
                    }
                    Err(e) => format!("Error: {}", e),
                }
            }
            "FileWrite" => {
                // Expect "path|content" in tool_arg
                let write_parts: Vec<&str> = tool_arg.splitn(2, '|').collect();
                if write_parts.len() < 2 {
                    "Error: FileWrite expects path|content".to_string()
                } else {
                    let path = write_parts[0].trim();
                    let content = write_parts[1];
                    let policy = self.cortex.policy();
                    let validated_path = match policy.validate_write_path(path) {
                        Ok(p) => p,
                        Err(e) => return format!("Security Error: {}", e),
                    };
                    match std::fs::write(&validated_path, content) {
                        Ok(()) => format!("Written to {}", validated_path.display()),
                        Err(e) => format!("Error: {}", e),
                    }
                }
            }
            _ => format!("Unknown tool: {}", tool_name),
        }
    }

    /// Run the autonomous cognitive loop on a given task.
    pub fn run(&mut self, persona: &str, user_task: &str) -> Result<String, String> {
        println!("╔══════════════════════════════════════════════════╗");
        println!("║   Initiating Sovereign Cognitive Loop...         ║");
        println!("╚══════════════════════════════════════════════════╝");

        let mut env = Environment::new();
        env.add_template("system", include_str!("../templates/system.jinja"))
            .map_err(|e| format!("Failed to load template: {}", e))?;

        let capabilities = vec![
            CapDesc { id: "ShellRunner".into(), description: "Run a shell command".into() },
            CapDesc { id: "FileRead".into(), description: "Read a file".into() },
            CapDesc { id: "FileWrite".into(), description: "Write a file".into() },
            CapDesc { id: "DirList".into(), description: "List directory contents".into() },
            CapDesc { id: "GitStatus".into(), description: "Get git repository status".into() },
            CapDesc { id: "CargoCheck".into(), description: "Verify Rust compilation".into() },
        ];

        let system_prompt = env.get_template("system")
            .map_err(|e| format!("Template get err: {}", e))?
            .render(context! {
                persona => persona,
                capabilities => capabilities
            })
            .map_err(|e| format!("Template render err: {}", e))?;

        // ── Qwen Chat Template ──
        // <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt.trim(),
            user_task
        );

        let prompt_ids = self.tokenizer.encode(&prompt)?;
        println!("[AGENT] Prompt: {} tokens", prompt_ids.len());
        
        use std::io::Write;

        // 1. Prefill
        if prompt_ids.is_empty() {
            return Err("Empty prompt".into());
        }

        let prefill_start = std::time::Instant::now();
        for i in 0..prompt_ids.len() - 1 {
            self.engine.step(prompt_ids[i])?;
        }
        let prefill_ms = prefill_start.elapsed().as_millis();
        let prefill_tps = if prefill_ms > 0 {
            (prompt_ids.len() - 1) as f64 / (prefill_ms as f64 / 1000.0)
        } else { 0.0 };
        println!("[PERF] Prefill: {} tokens in {}ms ({:.1} tok/s)",
            prompt_ids.len() - 1, prefill_ms, prefill_tps);

        let mut last_token = *prompt_ids.last().unwrap();
        let mut active_thought = String::new();
        let mut full_trajectory = String::new();
        let mut tool_calls = 0;
        let eos_id = self.engine.config.eos_token_id;
        let im_end_id = self.engine.config.im_end_token_id;
        let mut inside_think = false;
        let decode_start = std::time::Instant::now();
        let mut decode_tokens = 0u32;

        // 2. Continuous generation & reflection cycle
        for _step in 0..self.engine.config.max_tokens {
            // Check for interrupts non-blockingly
            if let Ok(msg) = self.interrupt_rx.try_recv() {
                println!("\n\n[HITL INTERRUPT] User injected: {}\n", msg);
                let interrupt_text = format!("<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", msg);
                let interrupt_ids = self.tokenizer.encode(&interrupt_text)?;
                if !interrupt_ids.is_empty() {
                    for i in 0..interrupt_ids.len() - 1 {
                        self.engine.step(interrupt_ids[i])?;
                    }
                    last_token = *interrupt_ids.last().unwrap();
                }
                active_thought.clear();
            }

            let next_token = self.engine.step(last_token)?;
            last_token = next_token;  // MUST be set before any continue!
            decode_tokens += 1;

            // EOS or <|im_end|>
            if next_token == eos_id || next_token == im_end_id {
                let decode_ms = decode_start.elapsed().as_millis();
                let decode_tps = if decode_ms > 0 {
                    decode_tokens as f64 / (decode_ms as f64 / 1000.0)
                } else { 0.0 };
                println!("\n[PERF] Decode: {} tokens in {}ms ({:.1} tok/s)",
                    decode_tokens, decode_ms, decode_tps);
                println!("[AGENT] ── Generation Complete ──");
                break;
            }

            let text = self.tokenizer.decode(&[next_token])?;

            // ── <think></think> tag handling (System 2 reasoning) ──
            active_thought.push_str(&text);
            full_trajectory.push_str(&text);

            if active_thought.contains("<think>") && !inside_think {
                inside_think = true;
                print!("\n[REASONING] ");
                std::io::stdout().flush().ok();
                continue;
            }

            if inside_think {
                if active_thought.contains("</think>") {
                    inside_think = false;
                    println!("\n[/REASONING]");
                    active_thought.clear();
                    continue;
                }
                print!("{}", text);
                std::io::stdout().flush().ok();
                continue;
            }

            print!("{}", text);
            std::io::stdout().flush().ok();

            // ── Tool call parsing ──
            if active_thought.contains("</call>") {
                tool_calls += 1;
                
                if tool_calls > self.max_reflection_steps {
                    println!("\n[CORTEX] 🚨 MAX TOOL CALLS REACHED ({}). Forcing answer. 🚨", self.max_reflection_steps);
                    let override_text = "\nI've reached the tool call limit. Let me give my final answer based on what I know.\n\n";
                    let obs_ids = self.tokenizer.encode(override_text)?;
                    for (i, &obs_id) in obs_ids.iter().enumerate() {
                        if i < obs_ids.len() - 1 {
                            self.engine.step(obs_id)?;
                        } else {
                            last_token = obs_id;
                        }
                    }
                    full_trajectory.push_str(override_text);
                    active_thought.clear();
                    continue;
                }

                // Extract tool call: <call>ToolName|arg</call>
                if let (Some(start_idx), Some(end_idx)) = (active_thought.rfind("<call>"), active_thought.rfind("</call>")) {
                    let call_content = &active_thought[start_idx + 6 .. end_idx];
                    let call_content = call_content.trim().to_string();
                    
                    println!("\n[CORTEX] ⚡ Executing: {}", call_content);
                    
                    // Actually execute the tool!
                    let observation = self.execute_tool(&call_content);
                    
                    println!("[CORTEX] Result: {}", if observation.len() > 100 { 
                        format!("{}...", &observation[..100]) 
                    } else { 
                        observation.clone() 
                    });

                    // Inject observation back into generation stream
                    let obs_text = format!("\nObservation: {}\n", observation);
                    print!("{}", obs_text);
                    std::io::stdout().flush().ok();

                    let obs_ids = self.tokenizer.encode(&obs_text)?;
                    for (i, &obs_id) in obs_ids.iter().enumerate() {
                        if i < obs_ids.len() - 1 {
                            self.engine.step(obs_id)?;
                        } else {
                            last_token = obs_id;
                        }
                    }
                    
                    full_trajectory.push_str(&obs_text);
                }
                
                active_thought.clear();
            }
        }
        
        Ok(full_trajectory)
    }
}
