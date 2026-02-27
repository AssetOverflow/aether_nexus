//! Agent Cognitive Loop (ReAct / Think / Execute)
//!
//! Ties the Cortex capabilities directly to the `InferenceEngine` streaming loop
//! to execute "Thought -> Action -> Observation" cycles autonomously.

use crate::cortex::Cortex;
use crate::inference::InferenceEngine;
use crate::tokenizer::Tokenizer;
use minijinja::{Environment, context};
use serde::Serialize;
use std::sync::mpsc::Receiver;

#[derive(Serialize)]
struct CapDesc {
    id: String,
    description: String,
}

pub struct AgentLoop<'a> {
    pub target_engine: &'a mut InferenceEngine,
    pub draft_engine: Option<&'a mut InferenceEngine>,
    pub tokenizer: &'a Tokenizer,
    pub cortex: &'a mut Cortex,
    pub interrupt_rx: Receiver<String>,
    pub max_reflection_steps: usize,
    /// Speculative draft token queue
    pub draft_queue: std::collections::VecDeque<u32>,
}

impl<'a> AgentLoop<'a> {
    pub fn new(
        target_engine: &'a mut InferenceEngine,
        draft_engine: Option<&'a mut InferenceEngine>,
        tokenizer: &'a Tokenizer,
        cortex: &'a mut Cortex,
        interrupt_rx: Receiver<String>,
        max_reflection_steps: usize,
    ) -> Self {
        Self {
            target_engine,
            draft_engine,
            tokenizer,
            cortex,
            interrupt_rx,
            max_reflection_steps,
            draft_queue: std::collections::VecDeque::new(),
        }
    }

    /// Execute a text-based tool call and return the observation string.
    pub(crate) fn execute_tool(&self, tool_call: &str) -> String {
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
                        let result = if !stdout.is_empty() {
                            stdout.to_string()
                        } else {
                            stderr.to_string()
                        };
                        // Truncate to avoid overwhelming the context (approx 500 chars)
                        let result_len = result.chars().count();
                        if result_len > 500 {
                            format!(
                                "{}...[truncated]",
                                result.chars().take(500).collect::<String>()
                            )
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
                        if content.chars().count() > 500 {
                            format!(
                                "{}...[truncated]",
                                content.chars().take(500).collect::<String>()
                            )
                        } else {
                            content
                        }
                    }
                    Err(e) => format!("Error reading '{}': {}", validated_path.display(), e),
                }
            }
            "DirList" | "FileList" => {
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
                        if result.is_empty() {
                            "clean - no changes".to_string()
                        } else {
                            result
                        }
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
                        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                        if stderr.chars().count() > 500 {
                            format!(
                                "{}...[truncated]",
                                stderr.chars().take(500).collect::<String>()
                            )
                        } else {
                            stderr
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
        crate::nexus_debug!("Initiating Sovereign Cognitive Loop...");

        let mut env = Environment::new();
        env.add_template("system", include_str!("../templates/system.jinja"))
            .map_err(|e| format!("Failed to load template: {}", e))?;

        let capabilities = vec![
            CapDesc {
                id: "ShellRunner".into(),
                description: "Run a shell command".into(),
            },
            CapDesc {
                id: "FileRead".into(),
                description: "Read a file".into(),
            },
            CapDesc {
                id: "FileWrite".into(),
                description: "Write a file".into(),
            },
            CapDesc {
                id: "FileList".into(),
                description: "List directory contents".into(),
            },
            CapDesc {
                id: "GitStatus".into(),
                description: "Get git repository status".into(),
            },
            CapDesc {
                id: "CargoCheck".into(),
                description: "Verify Rust compilation".into(),
            },
        ];

        let workspace_roots: Vec<String> = self.cortex.policy().roots()
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        let system_prompt = env
            .get_template("system")
            .map_err(|e| format!("Template get err: {}", e))?
            .render(context! {
                persona => persona,
                capabilities => capabilities,
                workspace_roots => workspace_roots
            })
            .map_err(|e| format!("Template render err: {}", e))?;

        // <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{task}\n(You MUST use a tool first before answering)<|im_end|>\n<|im_start|>assistant\n<think>\n
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}\n(You MUST use a <call> tool first before answering)<|im_end|>\n<|im_start|>assistant\n<think>\n",
            system_prompt.trim(),
            user_task
        );

        let prompt_ids = self.tokenizer.encode(&prompt)?;
        crate::nexus_debug!("Prompt: {} tokens", prompt_ids.len());

        use std::io::Write;

        // 1. Prefill
        if prompt_ids.is_empty() {
            return Err("Empty prompt".into());
        }

        let prefill_start = std::time::Instant::now();
        for i in 0..prompt_ids.len() - 1 {
            self.target_engine.step(prompt_ids[i], None)?;
            if let Some(ref mut draft) = self.draft_engine {
                draft.step(prompt_ids[i], None)?;
            }
        }
        let prefill_ms = prefill_start.elapsed().as_millis();
        let prefill_tps = if prefill_ms > 0 {
            (prompt_ids.len() - 1) as f64 / (prefill_ms as f64 / 1000.0)
        } else {
            0.0
        };
        crate::nexus_debug!(
            "Prefill: {} tokens in {}ms ({:.1} tok/s)",
            prompt_ids.len() - 1,
            prefill_ms,
            prefill_tps
        );

        let mut last_token = *prompt_ids.last().unwrap();
        let mut active_thought = String::new();
        let mut full_trajectory = String::new();
        let mut tool_calls = 0;
        let eos_id = self.target_engine.config.eos_token_id;
        let im_end_id = self.target_engine.config.im_end_token_id;
        let mut inside_think = false;
        let decode_start = std::time::Instant::now();
        let mut decode_tokens = 0u32;

        // 2. Continuous generation & reflection cycle
        'macro_loop: for _step in 0..self.target_engine.config.max_tokens {
            // Check for interrupts non-blockingly
            if let Ok(msg) = self.interrupt_rx.try_recv() {
                crate::nexus_info!("[HITL INTERRUPT] User injected: {}", msg);
                let interrupt_text = format!(
                    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    msg
                );
                let interrupt_ids = self.tokenizer.encode(&interrupt_text)?;
                if !interrupt_ids.is_empty() {
                    for i in 0..interrupt_ids.len() - 1 {
                        self.target_engine.step(interrupt_ids[i], None)?;
                        if let Some(ref mut draft) = self.draft_engine {
                            draft.step(interrupt_ids[i], None)?;
                        }
                    }
                    last_token = *interrupt_ids.last().unwrap();
                }
                active_thought.clear();
            }

            // --- Grammar Constraints ---
            let mut allowed_tokens: Option<&[u32]> = None;
            let mut new_tokens = Vec::new();

            if let Some(ref mut draft) = self.draft_engine {
                let max_draft = 3;
                let mut draft_tokens = Vec::with_capacity(max_draft);
                let mut current_draft_input = last_token;
                
                for _ in 0..max_draft {
                    let d_tok = draft.step(current_draft_input, allowed_tokens)?;
                    draft_tokens.push(d_tok);
                    current_draft_input = d_tok;
                    if d_tok == eos_id || d_tok == im_end_id {
                        break;
                    }
                }
                
                let actual_drafts = draft_tokens.len();
                let mut target_inputs = Vec::with_capacity(actual_drafts + 1);
                target_inputs.push(last_token);
                target_inputs.extend_from_slice(&draft_tokens);
                
                let all_logits = self.target_engine.forward_batch(&target_inputs)?;
                let vocab = self.target_engine.config.vocab_size as usize;
                
                let mut accept_count = 0;
                for i in 0..actual_drafts {
                    let logit_offset = i * vocab;
                    let step_logits = &all_logits[logit_offset..logit_offset+vocab];
                    let mut step_logits_vec = step_logits.to_vec();
                    
                    if let Some(allowed) = allowed_tokens {
                        for j in 0..vocab {
                            if !allowed.contains(&(j as u32)) {
                                step_logits_vec[j] = f32::NEG_INFINITY;
                            }
                        }
                    }
                    
                    let target_sampled = crate::inference::sample_token(&step_logits_vec, self.target_engine.config.temperature);
                    
                    if target_sampled == draft_tokens[i] {
                        new_tokens.push(draft_tokens[i]);
                        accept_count += 1;
                    } else {
                        new_tokens.push(target_sampled);
                        break;
                    }
                }
                
                if accept_count == actual_drafts {
                    let logit_offset = actual_drafts * vocab;
                    let step_logits = &all_logits[logit_offset..logit_offset+vocab];
                    let mut step_logits_vec = step_logits.to_vec();
                    
                    if let Some(allowed) = allowed_tokens {
                        for j in 0..vocab {
                            if !allowed.contains(&(j as u32)) {
                                step_logits_vec[j] = f32::NEG_INFINITY;
                            }
                        }
                    }
                    
                    let bonus_token = crate::inference::sample_token(&step_logits_vec, self.target_engine.config.temperature);
                    new_tokens.push(bonus_token);
                    
                    if actual_drafts > 0 {
                        let _ = draft.step(*draft_tokens.last().unwrap(), None); // advance KV cache for bonus token
                    }
                }
                
                let correct_pos = self.target_engine.push_loom() - (actual_drafts as u32 + 1) + new_tokens.len() as u32;
                self.target_engine.pop_loom(correct_pos);
                draft.pop_loom(correct_pos);
            } else {
                let next_token = self.target_engine.step(last_token, allowed_tokens)?;
                new_tokens.push(next_token);
            }

            for &next_token in &new_tokens {
                last_token = next_token; // MUST be set before any continue!
                decode_tokens += 1;

                // EOS or <|im_end|>
                if next_token == eos_id || next_token == im_end_id {
                    let decode_ms = decode_start.elapsed().as_millis();
                    let decode_tps = if decode_ms > 0 {
                        decode_tokens as f64 / (decode_ms as f64 / 1000.0)
                    } else {
                        0.0
                    };
                    crate::nexus_debug!(
                        "Decode: {} tokens in {}ms ({:.1} tok/s)",
                        decode_tokens, decode_ms, decode_tps
                    );
                    
                    println!();
                    println!("── {} tokens in {:.1}s ({:.1} tok/s) ──", decode_tokens, decode_ms as f32 / 1000.0, decode_tps);
                    break 'macro_loop;
                }

                let text = self.tokenizer.decode(&[next_token])?;

                // ── <think></think> tag handling (System 2 reasoning) ──
                active_thought.push_str(&text);
                full_trajectory.push_str(&text);

                if active_thought.contains("<think>") && !inside_think {
                    inside_think = true;
                    if crate::logging::show_thinking() {
                        print!("\n💭 ");
                        std::io::stdout().flush().ok();
                    }
                    continue;
                }

                if inside_think {
                    if active_thought.contains("</think>") {
                        inside_think = false;
                        if crate::logging::show_thinking() {
                            println!();
                        }
                        active_thought.clear();
                        continue;
                    }
                    if crate::logging::show_thinking() {
                        print!("{}", text);
                        std::io::stdout().flush().ok();
                    }
                    continue;
                }

                if crate::logging::show_thinking() {
                    print!("{}", text);
                    std::io::stdout().flush().ok();
                }

                // ── Tool call parsing ──
                if active_thought.contains("</call>") {
                    // Break hallucination loops (0.5B model often gets stuck emitting </call> endlessly)
                    if !active_thought.contains("<call>") {
                        crate::nexus_debug!("Intercepted stray </call> hallucination. Breaking loop.");
                        active_thought.clear();
                        continue;
                    }

                    tool_calls += 1;

                    if tool_calls > self.max_reflection_steps {
                        println!(
                            "\n[CORTEX] 🚨 MAX TOOL CALLS REACHED ({}). Forcing answer. 🚨",
                            self.max_reflection_steps
                        );
                        let override_text = "\nI've reached the tool call limit. Let me give my final answer based on what I know.\n\n";
                        let obs_ids = self.tokenizer.encode(override_text)?;
                        for (i, &obs_id) in obs_ids.iter().enumerate() {
                            if i < obs_ids.len() - 1 {
                                self.target_engine.step(obs_id, None)?;
                                if let Some(ref mut draft) = self.draft_engine {
                                    draft.step(obs_id, None)?;
                                }
                            } else {
                                last_token = obs_id;
                            }
                        }
                        full_trajectory.push_str(override_text);
                        active_thought.clear();
                        continue 'macro_loop;
                    }

                    // Extract tool call: <call>ToolName|arg</call>
                    if let (Some(start_idx), Some(end_idx)) = (
                        active_thought.rfind("<call>"),
                        active_thought.rfind("</call>"),
                    ) {
                        let call_content = &active_thought[start_idx + 6..end_idx];
                        let call_content = call_content.trim().to_string();

                        let tool_start = std::time::Instant::now();
                        let observation = self.execute_tool(&call_content);
                        let tool_elapsed = tool_start.elapsed();

                        let summary = if observation.chars().count() > 30 {
                            format!("{}...", observation.chars().take(30).collect::<String>().replace("\n", " "))
                        } else {
                            observation.clone().replace("\n", " ")
                        };

                        println!("\n⚡ {} → {} ({:.2}s)", call_content, summary, tool_elapsed.as_secs_f32());
                        crate::nexus_debug!("Tool execution: {} Result: {}", call_content, observation);

                        // Inject observation back into generation stream
                        let obs_text = format!("\nObservation: {}\n", observation);
                        
                        let obs_ids = self.tokenizer.encode(&obs_text)?;
                        for (i, &obs_id) in obs_ids.iter().enumerate() {
                            if i < obs_ids.len() - 1 {
                                self.target_engine.step(obs_id, None)?;
                                if let Some(ref mut draft) = self.draft_engine {
                                    draft.step(obs_id, None)?;
                                }
                            } else {
                                last_token = obs_id;
                            }
                        }

                        full_trajectory.push_str(&obs_text);
                        full_trajectory.push_str("<obs_end>");
                    }

                    active_thought.clear();
                    continue 'macro_loop;
                }
            }
        }

        if !crate::logging::show_thinking() {
            let mut final_answer = String::new();
            if let Some(last_obs_idx) = full_trajectory.rfind("<obs_end>") {
                let text_after = &full_trajectory[last_obs_idx + 9..].trim();
                if !text_after.is_empty() {
                    final_answer = format!("🤖 {}\n", text_after);
                }
            } else {
                let text_after = full_trajectory.replace("<think>", "").replace("</think>", "");
                if !text_after.trim().is_empty() {
                    final_answer = format!("🤖 {}\n", text_after.trim());
                }
            }
            
            if !final_answer.is_empty() {
                println!("\n{}", final_answer);
            }
        }

        Ok(full_trajectory)
    }
}
