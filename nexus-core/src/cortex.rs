//! The Unified Capability Cortex – typed, zero-copy dispatch engine.
//!
//! The Cortex is the bridge that renders the organism alive. It receives
//! action tensors from the Weaver decode kernel and dispatches them to
//! native Rust capability implementations.
//!
//! # Dispatch Protocol
//!
//! The action tensor contains three u32 values:
//! - `data[0]`: CapabilityId (which capability to invoke)
//! - `data[1]`: arg_offset (byte offset into observation buffer for arguments)
//! - `data[2]`: result_offset (byte offset for writing results)
//!
//! The Cortex reads arguments from the observation buffer, executes the
//! capability, and writes results back — all within the Fabric's pre-allocated
//! observation region.

use std::collections::HashMap;
use std::sync::Arc;

use crate::capability::{Capability, HandlerFn};
use crate::fabric::Fabric;
use crate::register_capability;
use crate::sandbox::SandboxPolicy;
use crate::types::{CapabilityId, CortexError, MAX_RESULT_SIZE, ModelDims};

use bytemuck::{Pod, Zeroable};

// ─────────────────────────────────────────────────────────────────────────────
// Built-in Capability Implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Arguments for the CargoCheck capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CargoCheckArgs {
    /// Offset into observation buffer where workspace path is stored (null-terminated)
    pub workspace_path_offset: u32,
    /// Compilation flags (reserved)
    pub flags: u32,
}

/// Result of the CargoCheck capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CargoCheckResult {
    /// Whether compilation succeeded
    pub success: u32, // 1 = success, 0 = failure
    /// Number of errors
    pub error_count: u32,
    /// Number of warnings
    pub warning_count: u32,
    /// Reserved padding
    pub _pad: u32,
}

/// CargoCheck: invoke `cargo check` via subprocess (air-gapped, local toolchain)
pub struct CargoCheck;

impl Capability for CargoCheck {
    type Args = CargoCheckArgs;
    type Result = CargoCheckResult;
    const ID: CapabilityId = CapabilityId::CargoCheck;

    fn execute(
        args: Self::Args,
        arg_buf: &[u8],
        _res_buf: &mut [u8],
        policy: &SandboxPolicy,
    ) -> Result<Self::Result, CortexError> {
        // Read workspace path from the observation buffer
        let path_offset = args.workspace_path_offset as usize;
        let path_end = arg_buf[path_offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| path_offset + p)
            .unwrap_or(arg_buf.len().min(path_offset + 256));

        let workspace_path = std::str::from_utf8(&arg_buf[path_offset..path_end])
            .map_err(|e| CortexError::ExecutionFailed(format!("invalid UTF-8 in path: {}", e)))?;

        // Validate workspace path against sandbox
        let validated_path = policy.validate_path(workspace_path)?;

        // Validate command against sandbox
        policy.validate_command("cargo check")?;

        // Invoke cargo check as subprocess (respects air-gap: no network)
        let output = std::process::Command::new("cargo")
            .args(["check", "--message-format=json"])
            .current_dir(&validated_path)
            .output()
            .map_err(|e| CortexError::Io(e))?;

        let success = output.status.success();
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Count errors and warnings from stderr
        let error_count = stderr.matches("error[").count() as u32;
        let warning_count = stderr.matches("warning:").count() as u32;

        Ok(CargoCheckResult {
            success: if success { 1 } else { 0 },
            error_count,
            warning_count,
            _pad: 0,
        })
    }
}

/// Arguments for VectorSearch capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VectorSearchArgs {
    /// Offset to query vector in observation buffer
    pub query_offset: u32,
    /// Dimension of query vector
    pub query_dim: u32,
    /// Maximum results to return
    pub top_k: u32,
    /// Reserved
    pub _pad: u32,
}

/// Result of VectorSearch capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VectorSearchResult {
    /// Number of results found
    pub result_count: u32,
    /// Offset where result indices are written
    pub indices_offset: u32,
    /// Offset where result scores are written
    pub scores_offset: u32,
    /// Reserved
    pub _pad: u32,
}

/// VectorSearch: semantic search via MLX matmul (placeholder)
pub struct VectorSearch;

impl Capability for VectorSearch {
    type Args = VectorSearchArgs;
    type Result = VectorSearchResult;
    const ID: CapabilityId = CapabilityId::VectorSearch;

    fn execute(
        _args: Self::Args,
        _arg_buf: &[u8],
        _res_buf: &mut [u8],
        _policy: &SandboxPolicy,
    ) -> Result<Self::Result, CortexError> {
        // TODO: Implement via MLX matmul against memory vectors
        Ok(VectorSearchResult {
            result_count: 0,
            indices_offset: 0,
            scores_offset: 0,
            _pad: 0,
        })
    }
}

/// Arguments for GitStatus capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GitStatusArgs {
    /// Offset to repo path in observation buffer (null-terminated)
    pub repo_path_offset: u32,
    /// Flags (reserved)
    pub flags: u32,
}

/// Result of GitStatus capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GitStatusResult {
    /// Number of modified files
    pub modified_count: u32,
    /// Number of new (untracked) files
    pub new_count: u32,
    /// Number of deleted files
    pub deleted_count: u32,
    /// Whether HEAD is detached
    pub detached: u32,
}

/// GitStatus: query git repository status via libgit2
pub struct GitStatus;

impl Capability for GitStatus {
    type Args = GitStatusArgs;
    type Result = GitStatusResult;
    const ID: CapabilityId = CapabilityId::GitStatus;

    fn execute(
        args: Self::Args,
        arg_buf: &[u8],
        _res_buf: &mut [u8],
        _policy: &SandboxPolicy,
    ) -> Result<Self::Result, CortexError> {
        let path_offset = args.repo_path_offset as usize;
        let path_end = arg_buf[path_offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| path_offset + p)
            .unwrap_or(arg_buf.len().min(path_offset + 256));

        let repo_path = std::str::from_utf8(&arg_buf[path_offset..path_end])
            .map_err(|e| CortexError::ExecutionFailed(format!("invalid UTF-8: {}", e)))?;

        let repo = git2::Repository::open(repo_path)
            .map_err(|e| CortexError::ExecutionFailed(format!("git open: {}", e)))?;

        let statuses = repo
            .statuses(None)
            .map_err(|e| CortexError::ExecutionFailed(format!("git status: {}", e)))?;

        let mut modified = 0u32;
        let mut new = 0u32;
        let mut deleted = 0u32;

        for entry in statuses.iter() {
            let status = entry.status();
            if status.intersects(
                git2::Status::WT_MODIFIED | git2::Status::INDEX_MODIFIED | git2::Status::WT_RENAMED,
            ) {
                modified += 1;
            }
            if status.intersects(git2::Status::WT_NEW | git2::Status::INDEX_NEW) {
                new += 1;
            }
            if status.intersects(git2::Status::WT_DELETED | git2::Status::INDEX_DELETED) {
                deleted += 1;
            }
        }

        let detached = repo.head_detached().unwrap_or(false);

        Ok(GitStatusResult {
            modified_count: modified,
            new_count: new,
            deleted_count: deleted,
            detached: if detached { 1 } else { 0 },
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorRegex Capability (regex-powered text search via grep-* crates)
// ─────────────────────────────────────────────────────────────────────────────

/// Arguments for TensorRegex capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TensorRegexArgs {
    /// Offset to regex pattern string in observation buffer (null-terminated)
    pub pattern_offset: u32,
    /// Offset to search text in observation buffer (null-terminated)
    pub text_offset: u32,
    /// Maximum matches to return
    pub max_matches: u32,
    /// Reserved
    pub _pad: u32,
}

/// Result of TensorRegex capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TensorRegexResult {
    /// Number of matches found
    pub match_count: u32,
    /// Offset where match data is written in result buffer
    pub matches_offset: u32,
}

/// TensorRegex: regex-powered text search using the grep-regex crate.
pub struct TensorRegex;

impl Capability for TensorRegex {
    type Args = TensorRegexArgs;
    type Result = TensorRegexResult;
    const ID: CapabilityId = CapabilityId::TensorRegex;

    fn execute(
        args: Self::Args,
        arg_buf: &[u8],
        res_buf: &mut [u8],
        _policy: &SandboxPolicy,
    ) -> Result<Self::Result, CortexError> {
        use grep_matcher::Matcher;
        use grep_regex::RegexMatcher;

        let pattern_offset = args.pattern_offset as usize;
        let pattern_end = arg_buf[pattern_offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| pattern_offset + p)
            .unwrap_or(arg_buf.len().min(pattern_offset + 256));

        let pattern = std::str::from_utf8(&arg_buf[pattern_offset..pattern_end]).map_err(|e| {
            CortexError::ExecutionFailed(format!("invalid UTF-8 in pattern: {}", e))
        })?;

        let text_offset = args.text_offset as usize;
        let text_end = arg_buf[text_offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| text_offset + p)
            .unwrap_or(arg_buf.len().min(text_offset + 4096));

        let text = std::str::from_utf8(&arg_buf[text_offset..text_end])
            .map_err(|e| CortexError::ExecutionFailed(format!("invalid UTF-8 in text: {}", e)))?;

        let matcher = RegexMatcher::new(pattern)
            .map_err(|e| CortexError::ExecutionFailed(format!("invalid regex: {}", e)))?;

        let max_matches = args.max_matches.max(1).min(256) as usize;
        let mut match_count = 0u32;
        let result_struct_size = std::mem::size_of::<TensorRegexResult>();
        let mut write_offset = result_struct_size;

        // Find matches and write byte offset pairs (start, end) as u32s
        let haystack = text.as_bytes();
        let mut start_pos = 0;

        while (match_count as usize) < max_matches && start_pos < haystack.len() {
            match matcher.find_at(haystack, start_pos) {
                Ok(Some(m)) => {
                    // Write (start, end) as two u32s
                    let match_start = m.start() as u32;
                    let match_end = m.end() as u32;

                    if write_offset + 8 <= res_buf.len() {
                        res_buf[write_offset..write_offset + 4]
                            .copy_from_slice(&match_start.to_le_bytes());
                        res_buf[write_offset + 4..write_offset + 8]
                            .copy_from_slice(&match_end.to_le_bytes());
                        write_offset += 8;
                    }
                    match_count += 1;
                    start_pos = m.end().max(start_pos + 1);
                }
                Ok(None) => break,
                Err(e) => {
                    return Err(CortexError::ExecutionFailed(format!(
                        "regex search error: {}",
                        e
                    )));
                }
            }
        }

        Ok(TensorRegexResult {
            match_count,
            matches_offset: result_struct_size as u32,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CortexFS Capabilities
// ─────────────────────────────────────────────────────────────────────────────

/// Arguments for FileRead capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FileReadArgs {
    pub path_offset: u32,
    pub max_bytes: u32,
}

/// Result of FileRead capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FileReadResult {
    pub bytes_read: u32,
    pub data_offset: u32,
}

use wasmtime::{Engine, Instance, Module, Store};

pub struct FileRead;

impl Capability for FileRead {
    type Args = FileReadArgs;
    type Result = FileReadResult;
    const ID: CapabilityId = CapabilityId::FileRead;

    fn execute(
        args: Self::Args,
        arg_buf: &[u8],
        res_buf: &mut [u8],
        policy: &SandboxPolicy,
    ) -> Result<Self::Result, CortexError> {
        let path_offset = args.path_offset as usize;
        let path_end = arg_buf[path_offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| path_offset + p)
            .unwrap_or(arg_buf.len().min(path_offset + 256));

        let path_str = std::str::from_utf8(&arg_buf[path_offset..path_end])
            .map_err(|e| CortexError::ExecutionFailed(format!("invalid UTF-8 in path: {}", e)))?;

        // Validate path against sandbox
        let validated_path = policy.validate_path(path_str)?;

        let data = std::fs::read(&validated_path).map_err(CortexError::Io)?;

        let result_struct_size = std::mem::size_of::<FileReadResult>();
        let max_copy =
            (args.max_bytes as usize).min(res_buf.len().saturating_sub(result_struct_size));
        let to_copy = data.len().min(max_copy);

        if to_copy > 0 {
            res_buf[result_struct_size..result_struct_size + to_copy]
                .copy_from_slice(&data[..to_copy]);
        }

        Ok(FileReadResult {
            bytes_read: to_copy as u32,
            data_offset: result_struct_size as u32,
        })
    }
}

/// Arguments for FileWrite capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FileWriteArgs {
    pub path_offset: u32,
    pub data_offset: u32,
    pub data_len: u32,
    pub _pad: u32,
}

/// Result of FileWrite capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FileWriteResult {
    pub success: u32,
    pub bytes_written: u32,
}

pub struct FileWrite;

impl Capability for FileWrite {
    type Args = FileWriteArgs;
    type Result = FileWriteResult;
    const ID: CapabilityId = CapabilityId::FileWrite;

    fn execute(
        args: Self::Args,
        arg_buf: &[u8],
        _res_buf: &mut [u8],
        policy: &SandboxPolicy,
    ) -> Result<Self::Result, CortexError> {
        let path_offset = args.path_offset as usize;
        let path_end = arg_buf[path_offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| path_offset + p)
            .unwrap_or(arg_buf.len().min(path_offset + 256));

        let path_str = std::str::from_utf8(&arg_buf[path_offset..path_end])
            .map_err(|e| CortexError::ExecutionFailed(format!("invalid UTF-8 in path: {}", e)))?;

        // Validate write path against sandbox
        let validated_path = policy.validate_write_path(path_str)?;

        let data_offset = args.data_offset as usize;
        let data_len = args.data_len as usize;

        if data_offset + data_len > arg_buf.len() {
            return Err(CortexError::ArgOutOfBounds {
                offset: data_offset,
                size: arg_buf.len(),
            });
        }

        let data = &arg_buf[data_offset..data_offset + data_len];
        std::fs::write(&validated_path, data).map_err(CortexError::Io)?;

        Ok(FileWriteResult {
            success: 1,
            bytes_written: data_len as u32,
        })
    }
}

/// Arguments for DirList capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DirListArgs {
    pub path_offset: u32,
    pub _pad: u32,
}

/// Result of DirList capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DirListResult {
    pub entries_count: u32,
    pub text_offset: u32,
}

pub struct DirList;

impl Capability for DirList {
    type Args = DirListArgs;
    type Result = DirListResult;
    const ID: CapabilityId = CapabilityId::DirList;

    fn execute(
        args: Self::Args,
        arg_buf: &[u8],
        res_buf: &mut [u8],
        policy: &SandboxPolicy,
    ) -> Result<Self::Result, CortexError> {
        let path_offset = args.path_offset as usize;
        let path_end = arg_buf[path_offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| path_offset + p)
            .unwrap_or(arg_buf.len().min(path_offset + 256));

        let path_str = std::str::from_utf8(&arg_buf[path_offset..path_end])
            .map_err(|e| CortexError::ExecutionFailed(format!("invalid UTF-8 in path: {}", e)))?;

        // Validate path against sandbox
        let validated_path = policy.validate_path(path_str)?;

        let entries = std::fs::read_dir(&validated_path).map_err(CortexError::Io)?;
        let mut count = 0;
        let mut listing = String::new();

        for entry in entries {
            if let Ok(entry) = entry {
                let name = entry.file_name();
                listing.push_str(&name.to_string_lossy());
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        listing.push('/');
                    }
                }
                listing.push('\n');
                count += 1;
            }
        }

        let result_struct_size = std::mem::size_of::<DirListResult>();
        let data_bytes = listing.as_bytes();
        let max_copy = res_buf.len().saturating_sub(result_struct_size);
        let to_copy = data_bytes.len().min(max_copy);

        if to_copy > 0 {
            res_buf[result_struct_size..result_struct_size + to_copy]
                .copy_from_slice(&data_bytes[..to_copy]);
        }

        Ok(DirListResult {
            entries_count: count as u32,
            text_offset: result_struct_size as u32,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ShellRunner Capability
// ─────────────────────────────────────────────────────────────────────────────

/// Arguments for ShellRunner capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShellRunnerArgs {
    pub workspace_path_offset: u32,
    pub cmd_offset: u32,
    pub timeout_ms: u32,
    pub _pad: u32,
}

/// Result of ShellRunner capability
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShellRunnerResult {
    pub exit_code: u32,
    pub stdout_offset: u32,
    pub stderr_offset: u32,
    pub _pad: u32,
}

pub struct ShellRunner;

impl Capability for ShellRunner {
    type Args = ShellRunnerArgs;
    type Result = ShellRunnerResult;
    const ID: CapabilityId = CapabilityId::ShellRunner;

    fn execute(
        args: Self::Args,
        arg_buf: &[u8],
        res_buf: &mut [u8],
        policy: &SandboxPolicy,
    ) -> Result<Self::Result, CortexError> {
        let path_offset = args.workspace_path_offset as usize;
        let path_end = arg_buf[path_offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| path_offset + p)
            .unwrap_or(arg_buf.len().min(path_offset + 256));

        let workspace_path = std::str::from_utf8(&arg_buf[path_offset..path_end]).map_err(|e| {
            CortexError::ExecutionFailed(format!("invalid UTF-8 in workspace_path: {}", e))
        })?;

        // Validate workspace path against sandbox
        let validated_workspace = policy.validate_path(workspace_path)?;

        let cmd_offset = args.cmd_offset as usize;
        let cmd_end = arg_buf[cmd_offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| cmd_offset + p)
            .unwrap_or(arg_buf.len().min(cmd_offset + 1024));

        let cmd_str = std::str::from_utf8(&arg_buf[cmd_offset..cmd_end])
            .map_err(|e| CortexError::ExecutionFailed(format!("invalid UTF-8 in cmd: {}", e)))?;

        // Validate and parse command against sandbox policy (no more raw sh -c!)
        let (binary, cmd_args) = policy.validate_command(cmd_str)?;

        // Use the configured timeout, capped by policy max
        let timeout = if args.timeout_ms > 0 {
            std::time::Duration::from_millis(
                (args.timeout_ms as u64).min(policy.max_timeout.as_millis() as u64),
            )
        } else {
            policy.max_timeout
        };

        // Execute with structured command (no shell=True equivalent)
        let mut command = std::process::Command::new(&binary);
        if policy.clear_env {
            command.env_clear();
            if let Ok(path) = std::env::var("PATH") {
                command.env("PATH", path);
            }
        }

        let child = command
            .args(&cmd_args)
            .current_dir(&validated_workspace)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| CortexError::Io(e))?;

        // Enforce timeout using a thread that waits for the output.
        // We move the child into a thread and use a channel to get the result back.
        let (tx, rx) = std::sync::mpsc::channel();
        let timeout_thread = std::thread::spawn(move || {
            let output = child.wait_with_output();
            let _ = tx.send(output);
        });

        match rx.recv_timeout(timeout) {
            Ok(Ok(output)) => {
                let _ = timeout_thread.join();

                let exit_code = output.status.code().unwrap_or(1) as u32;

                let result_struct_size = std::mem::size_of::<ShellRunnerResult>();
                let mut curr_offset = result_struct_size;

                let max_stdout = res_buf.len().saturating_sub(curr_offset);
                let stdout_copy = output.stdout.len().min(max_stdout);

                if stdout_copy > 0 {
                    res_buf[curr_offset..curr_offset + stdout_copy]
                        .copy_from_slice(&output.stdout[..stdout_copy]);
                }
                let stdout_offset = curr_offset as u32;
                curr_offset += stdout_copy;

                let max_stderr = res_buf.len().saturating_sub(curr_offset);
                let stderr_copy = output.stderr.len().min(max_stderr);

                if stderr_copy > 0 {
                    res_buf[curr_offset..curr_offset + stderr_copy]
                        .copy_from_slice(&output.stderr[..stderr_copy]);
                }
                let stderr_offset = curr_offset as u32;

                Ok(ShellRunnerResult {
                    exit_code,
                    stdout_offset,
                    stderr_offset,
                    _pad: 0,
                })
            }
            Ok(Err(e)) => Err(CortexError::Io(e)),
            Err(_) => {
                // Timeout — the thread is still running with the child.
                // We can't kill from here since child was moved, but the thread
                // will eventually finish. Log the timeout.
                Err(CortexError::ExecutionFailed(format!(
                    "Command '{}' timed out after {} ms",
                    binary,
                    timeout.as_millis()
                )))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SafeEval (Wasmtime sandbox with fuel + timeout enforcement)
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct SafeEvalArgs {
    pub wasm_offset: u32,
    pub wasm_len: u32,
    pub timeout_ms: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct SafeEvalResult {
    pub exit_code: u32,
    pub stdout_offset: u32,
    pub stderr_offset: u32,
    pub _pad: u32,
}

pub struct SafeEval;

impl Capability for SafeEval {
    type Args = SafeEvalArgs;
    type Result = SafeEvalResult;
    const ID: CapabilityId = CapabilityId::SafeEval;

    fn execute(
        args: Self::Args,
        arg_buf: &[u8],
        res_buf: &mut [u8],
        policy: &SandboxPolicy,
    ) -> Result<Self::Result, CortexError> {
        let offset = args.wasm_offset as usize;
        let len = args.wasm_len as usize;

        if offset + len > arg_buf.len() {
            return Err(CortexError::ArgOutOfBounds {
                offset,
                size: arg_buf.len(),
            });
        }

        let wasm_bytes = &arg_buf[offset..offset + len];

        crate::nexus_debug!(
            "[CORTEX] SafeEval: executing {} bytes of WebAssembly",
            len
        );
        crate::nexus_debug!(
            "[CORTEX] Wasmtime fuel: {}, memory pages: {}",
            policy.wasm_fuel,
            policy.wasm_memory_pages
        );

        // Create engine with fuel metering enabled
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        config.epoch_interruption(true);

        let engine = Engine::new(&config)
            .map_err(|e| CortexError::ExecutionFailed(format!("Wasmtime engine: {}", e)))?;

        let module = match Module::new(&engine, wasm_bytes) {
            Ok(m) => m,
            Err(e) => {
                return Err(CortexError::ExecutionFailed(format!(
                    "Invalid Wasm module: {}",
                    e
                )));
            }
        };

        let mut store = Store::new(&engine, ());

        // Set fuel limit from sandbox policy
        store
            .set_fuel(policy.wasm_fuel)
            .map_err(|e| CortexError::ExecutionFailed(format!("Failed to set fuel: {}", e)))?;

        // Enable epoch-based interruption for timeout enforcement
        store.epoch_deadline_trap();
        store.set_epoch_deadline(1);

        // Start a background thread to increment the epoch after timeout
        let engine_clone = engine.clone();
        let timeout_ms = if args.timeout_ms > 0 {
            (args.timeout_ms as u64).min(policy.max_timeout.as_millis() as u64)
        } else {
            policy.max_timeout.as_millis() as u64
        };

        let timeout_handle = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(timeout_ms));
            engine_clone.increment_epoch();
        });

        let instance = match Instance::new(&mut store, &module, &[]) {
            Ok(i) => i,
            Err(e) => {
                return Err(CortexError::ExecutionFailed(format!(
                    "Wasm instantiation failed: {}",
                    e
                )));
            }
        };

        // Expect the module to export a `run` function returning an i32 exit code
        let typed_func = match instance.get_typed_func::<(), i32>(&mut store, "run") {
            Ok(f) => f,
            Err(e) => {
                return Err(CortexError::ExecutionFailed(format!(
                    "Wasm missing 'run' fn returning i32: {}",
                    e
                )));
            }
        };

        let exit_code = match typed_func.call(&mut store, ()) {
            Ok(c) => c,
            Err(e) => {
                // Check if this was a fuel exhaustion or epoch interruption
                if e.to_string().contains("fuel") {
                    return Err(CortexError::ExecutionFailed(format!(
                        "Wasm execution exceeded fuel limit ({})",
                        policy.wasm_fuel
                    )));
                }
                if e.to_string().contains("epoch") {
                    return Err(CortexError::ExecutionFailed(format!(
                        "Wasm execution timed out after {} ms",
                        timeout_ms
                    )));
                }
                return Err(CortexError::ExecutionFailed(format!(
                    "Wasm execution trapped: {}",
                    e
                )));
            }
        };

        // Clean up timeout thread (it'll exit on its own after the sleep)
        let _ = timeout_handle;

        let result_struct_size = std::mem::size_of::<SafeEvalResult>();
        let remaining_fuel = store.get_fuel().unwrap_or(0);
        let stdout_msg = format!(
            "SafeEval OK, exit_code: {}, fuel_remaining: {}/{}\n",
            exit_code, remaining_fuel, policy.wasm_fuel
        );

        // Write stdout string
        let max_copy = res_buf.len().saturating_sub(result_struct_size);
        let to_copy = stdout_msg.len().min(max_copy);

        if to_copy > 0 {
            res_buf[result_struct_size..result_struct_size + to_copy]
                .copy_from_slice(&stdout_msg.as_bytes()[..to_copy]);
        }

        Ok(SafeEvalResult {
            exit_code: 0,
            stdout_offset: result_struct_size as u32,
            stderr_offset: 0,
            _pad: 0,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The Cortex
// ─────────────────────────────────────────────────────────────────────────────

/// The Unified Capability Cortex.
///
/// Holds a map from `CapabilityId` to type-erased handler functions,
/// and a shared `SandboxPolicy` that all handlers consult for security decisions.
pub struct Cortex {
    /// Dispatch table: CapabilityId → handler function
    handlers: HashMap<CapabilityId, HandlerFn>,
    /// Shared sandbox policy for all capabilities
    policy: Arc<SandboxPolicy>,
}

impl Cortex {
    /// Boot the Cortex, registering all built-in capabilities.
    ///
    /// The `SandboxPolicy` is shared via `Arc` to all capability handlers.
    pub fn boot(policy: SandboxPolicy) -> Self {
        let mut cortex = Cortex {
            handlers: HashMap::new(),
            policy: Arc::new(policy),
        };

        // Register all built-in capabilities (must match CapabilityId::COUNT)
        register_capability!(cortex, CargoCheck);
        register_capability!(cortex, VectorSearch);
        register_capability!(cortex, GitStatus);
        register_capability!(cortex, TensorRegex);
        register_capability!(cortex, SafeEval);
        register_capability!(cortex, FileRead);
        register_capability!(cortex, FileWrite);
        register_capability!(cortex, DirList);
        register_capability!(cortex, ShellRunner);

        // Compile-time guarantee: if this assert fires, someone added a CapabilityId
        // variant without registering a handler (or vice versa).
        debug_assert_eq!(
            cortex.handlers.len(),
            CapabilityId::COUNT,
            "CapabilityId::COUNT ({}) != registered handlers ({}). \
             Every CapabilityId variant MUST have a registered handler.",
            CapabilityId::COUNT,
            cortex.handlers.len()
        );

        cortex
    }

    /// Register a capability handler
    pub fn register(&mut self, id: CapabilityId, handler: HandlerFn) {
        self.handlers.insert(id, handler);
    }

    /// Dispatch an action to the appropriate capability.
    ///
    /// # Protocol
    ///
    /// The action data contains:
    /// - `action[0]`: CapabilityId (u32)
    /// - `action[1]`: arg_offset into observation buffer
    /// - `action[2]`: result_offset into observation buffer
    ///
    /// The handler reads args from `obs[arg_offset..]` and writes results
    /// to `obs[result_offset..]`. The sandbox policy is passed to the handler
    /// for security enforcement.
    pub fn dispatch<D: ModelDims>(
        &self,
        action_data: &[u32],
        fabric: &mut Fabric<D>,
    ) -> Result<(), CortexError> {
        if action_data.len() < 3 {
            return Err(CortexError::ExecutionFailed(
                "action tensor must contain at least 3 u32 values".into(),
            ));
        }

        let raw_id = action_data[0];
        let arg_off = action_data[1] as usize;
        let res_off = action_data[2] as usize;

        let id = CapabilityId::from_raw(raw_id).ok_or(CortexError::InvalidId(raw_id))?;

        let handler = self
            .handlers
            .get(&id)
            .ok_or(CortexError::InvalidId(raw_id))?;

        // Get mutable observation buffer
        let obs = fabric.observation_bufs_mut();

        // Validate offsets
        if arg_off >= obs.len() {
            return Err(CortexError::ArgOutOfBounds {
                offset: arg_off,
                size: obs.len(),
            });
        }
        if res_off >= obs.len() || res_off + MAX_RESULT_SIZE > obs.len() {
            return Err(CortexError::ResultOutOfBounds {
                offset: res_off,
                size: obs.len(),
            });
        }

        // Split the observation buffer into arg (read) and result (write) regions.
        // We require arg_off < res_off for non-overlapping access.
        if arg_off >= res_off {
            return Err(CortexError::ExecutionFailed(
                "arg_offset must be less than result_offset for safe dispatch".into(),
            ));
        }

        let (left, right) = obs.split_at_mut(res_off);
        let arg_slice = &left[arg_off..];
        let right_len = right.len();
        let res_slice = &mut right[..MAX_RESULT_SIZE.min(right_len)];

        handler(arg_slice, res_slice, &self.policy)
    }

    /// Get the number of registered capabilities
    pub fn capability_count(&self) -> usize {
        self.handlers.len()
    }

    /// Check if a capability is registered
    pub fn has_capability(&self, id: CapabilityId) -> bool {
        self.handlers.contains_key(&id)
    }

    /// Get a reference to the sandbox policy
    pub fn policy(&self) -> &SandboxPolicy {
        &self.policy
    }
}

impl std::fmt::Debug for Cortex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cortex")
            .field("capability_count", &self.handlers.len())
            .field("registered_ids", &self.handlers.keys().collect::<Vec<_>>())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::create_genesis;
    use crate::types::Llama8B;
    use ring::rand::SystemRandom;
    use ring::signature::Ed25519KeyPair;

    fn test_policy() -> SandboxPolicy {
        SandboxPolicy::dev_default()
    }

    /// Cortex boots with all capabilities registered, matching CapabilityId::COUNT
    #[test]
    fn cortex_boots_with_capabilities() {
        let cortex = Cortex::boot(test_policy());
        assert_eq!(
            cortex.capability_count(),
            CapabilityId::COUNT,
            "All CapabilityId variants must be registered"
        );
        assert!(cortex.has_capability(CapabilityId::CargoCheck));
        assert!(cortex.has_capability(CapabilityId::VectorSearch));
        assert!(cortex.has_capability(CapabilityId::GitStatus));
        assert!(cortex.has_capability(CapabilityId::TensorRegex));
        assert!(cortex.has_capability(CapabilityId::SafeEval));
        assert!(cortex.has_capability(CapabilityId::FileRead));
        assert!(cortex.has_capability(CapabilityId::FileWrite));
        assert!(cortex.has_capability(CapabilityId::DirList));
        assert!(cortex.has_capability(CapabilityId::ShellRunner));
    }

    /// Invalid capability ID is rejected
    #[test]
    fn cortex_rejects_invalid_id() {
        let cortex = Cortex::boot(test_policy());
        let tmp = std::env::temp_dir().join("test_cortex_invalid.aether");
        let rng = SystemRandom::new();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();

        create_genesis::<Llama8B>(tmp.to_str().unwrap(), &key_pair).unwrap();
        let mut fabric = Fabric::<Llama8B>::boot(tmp.to_str().unwrap()).unwrap();

        let action = [999u32, 0, 4096]; // invalid capability ID
        let result = cortex.dispatch(&action, &mut fabric);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&tmp);
    }

    /// CargoCheckArgs and CargoCheckResult are Pod-compatible
    #[test]
    fn cargo_check_types_are_pod() {
        let args = CargoCheckArgs {
            workspace_path_offset: 0,
            flags: 0,
        };
        let bytes = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), std::mem::size_of::<CargoCheckArgs>());

        let result = CargoCheckResult {
            success: 1,
            error_count: 0,
            warning_count: 0,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&result);
        assert_eq!(bytes.len(), std::mem::size_of::<CargoCheckResult>());
    }

    /// GitStatus types are Pod-compatible
    #[test]
    fn git_status_types_are_pod() {
        let args = GitStatusArgs {
            repo_path_offset: 0,
            flags: 0,
        };
        let _bytes = bytemuck::bytes_of(&args);

        let result = GitStatusResult {
            modified_count: 5,
            new_count: 2,
            deleted_count: 1,
            detached: 0,
        };
        let bytes = bytemuck::bytes_of(&result);
        let recovered: &GitStatusResult = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.modified_count, 5);
    }

    #[test]
    fn file_read_types_are_pod() {
        let args = FileReadArgs {
            path_offset: 0,
            max_bytes: 4096,
        };
        let bytes = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), std::mem::size_of::<FileReadArgs>());

        let result = FileReadResult {
            bytes_read: 10,
            data_offset: 0,
        };
        let bytes = bytemuck::bytes_of(&result);
        assert_eq!(bytes.len(), std::mem::size_of::<FileReadResult>());
    }

    #[test]
    fn shell_runner_types_are_pod() {
        let args = ShellRunnerArgs {
            workspace_path_offset: 0,
            cmd_offset: 0,
            timeout_ms: 5000,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), std::mem::size_of::<ShellRunnerArgs>());

        let result = ShellRunnerResult {
            exit_code: 0,
            stdout_offset: 0,
            stderr_offset: 0,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&result);
        assert_eq!(bytes.len(), std::mem::size_of::<ShellRunnerResult>());
    }

    #[test]
    fn safe_eval_types_are_pod() {
        let args = SafeEvalArgs {
            wasm_offset: 0,
            wasm_len: 0,
            timeout_ms: 1000,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), std::mem::size_of::<SafeEvalArgs>());

        let result = SafeEvalResult {
            exit_code: 0,
            stdout_offset: 0,
            stderr_offset: 0,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&result);
        assert_eq!(bytes.len(), std::mem::size_of::<SafeEvalResult>());
    }

    #[test]
    fn tensor_regex_types_are_pod() {
        let args = TensorRegexArgs {
            pattern_offset: 0,
            text_offset: 0,
            max_matches: 10,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), std::mem::size_of::<TensorRegexArgs>());

        let result = TensorRegexResult {
            match_count: 0,
            matches_offset: 0,
        };
        let bytes = bytemuck::bytes_of(&result);
        assert_eq!(bytes.len(), std::mem::size_of::<TensorRegexResult>());
    }
}
