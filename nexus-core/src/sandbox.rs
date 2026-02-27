//! SandboxPolicy – Single authority for all capability security decisions.
//!
//! Every capability that touches the filesystem, executes subprocesses, or runs
//! WebAssembly must consult the SandboxPolicy before performing the action.
//! This module is the **only** place where security enforcement logic lives.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::types::{CortexError, SecurityConfig};

/// The single security authority for all Cortex capabilities.
///
/// Constructed once at boot from `NexusConfig::security` and shared
/// via `Arc<SandboxPolicy>` to every capability handler.
#[derive(Debug)]
pub struct SandboxPolicy {
    /// Canonicalized workspace roots. All file ops must resolve within one of these.
    allowed_roots: Vec<PathBuf>,
    /// Allowed subprocess binary names (e.g. "cargo", "git", "ls").
    allowed_commands: HashSet<String>,
    /// Clear parent environment variables for subprocesses.
    pub clear_env: bool,
    /// Maximum execution timeout for subprocesses.
    pub max_timeout: Duration,
    /// Wasmtime fuel limit per execution.
    pub wasm_fuel: u64,
    /// Wasmtime memory page limit.
    pub wasm_memory_pages: u32,
}

impl SandboxPolicy {
    /// Construct a SandboxPolicy from configuration.
    ///
    /// Canonicalizes all workspace roots at construction time so that
    /// runtime path checks are fast and unambiguous.
    pub fn from_config(config: &SecurityConfig) -> Self {
        let allowed_roots: Vec<PathBuf> = config
            .workspace_roots
            .iter()
            .filter_map(|root| match std::fs::canonicalize(root) {
                Ok(canonical) => Some(canonical),
                Err(e) => {
                    eprintln!(
                        "[SANDBOX] Warning: cannot canonicalize workspace root '{}': {}",
                        root, e
                    );
                    None
                }
            })
            .collect();

        let allowed_commands: HashSet<String> = config.allowed_commands.iter().cloned().collect();

        Self {
            allowed_roots,
            allowed_commands,
            clear_env: config.clear_env,
            max_timeout: Duration::from_millis(config.max_subprocess_timeout_ms),
            wasm_fuel: config.wasm_fuel,
            wasm_memory_pages: config.wasm_memory_pages,
        }
    }

    /// Create a permissive default policy for development.
    ///
    /// Uses the current working directory as the sole workspace root
    /// and allows common development commands.
    pub fn dev_default() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let canonical_cwd = std::fs::canonicalize(&cwd).unwrap_or(cwd);

        let allowed_commands: HashSet<String> = [
            "cargo",
            "git",
            "ls",
            "cat",
            "head",
            "tail",
            "find",
            "grep",
            "wc",
            "echo",
            "mkdir",
            "touch",
            "cp",
            "mv",
            "rustfmt",
            "clippy-driver",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            allowed_roots: vec![canonical_cwd],
            allowed_commands,
            clear_env: true,
            max_timeout: Duration::from_secs(30),
            wasm_fuel: 1_000_000,
            wasm_memory_pages: 256, // 16 MB
        }
    }

    /// Validate and resolve a path against the sandbox.
    ///
    /// Returns the canonicalized path if it falls within an allowed workspace root.
    /// Rejects:
    /// - Absolute paths not rooted in an allowed workspace
    /// - Path traversal (`..` components that escape the workspace)
    /// - Symlinks that resolve outside the workspace
    pub fn validate_path(&self, path: &str) -> Result<PathBuf, CortexError> {
        if self.allowed_roots.is_empty() {
            return Err(CortexError::SandboxViolation(
                "No workspace roots configured — all file operations are denied".into(),
            ));
        }

        let raw_path = Path::new(path);

        // If path is relative, resolve against the first workspace root.
        // If absolute, use it directly but still validate against roots.
        let resolved = if raw_path.is_absolute() {
            raw_path.to_path_buf()
        } else {
            self.allowed_roots[0].join(raw_path)
        };

        // Canonicalize to resolve symlinks, `..`, and `.` components.
        // This is the critical step: after canonicalization, the path is
        // a real absolute path with no traversal tricks.
        let canonical = std::fs::canonicalize(&resolved).map_err(|e| {
            CortexError::SandboxViolation(format!(
                "Cannot resolve path '{}': {} (must exist and be accessible)",
                path, e
            ))
        })?;

        // Check that the canonical path starts with at least one allowed root.
        let is_within_workspace = self
            .allowed_roots
            .iter()
            .any(|root| canonical.starts_with(root));

        if !is_within_workspace {
            return Err(CortexError::SandboxViolation(format!(
                "Path '{}' resolves to '{}' which is outside all allowed workspace roots",
                path,
                canonical.display()
            )));
        }

        Ok(canonical)
    }

    /// Validate a path that may not exist yet (for write operations).
    ///
    /// Validates the parent directory instead of the full path.
    pub fn validate_write_path(&self, path: &str) -> Result<PathBuf, CortexError> {
        if self.allowed_roots.is_empty() {
            return Err(CortexError::SandboxViolation(
                "No workspace roots configured — all file operations are denied".into(),
            ));
        }

        let raw_path = Path::new(path);

        let resolved = if raw_path.is_absolute() {
            raw_path.to_path_buf()
        } else {
            self.allowed_roots[0].join(raw_path)
        };

        // For write paths, the file may not exist yet, so canonicalize the parent.
        let parent = resolved
            .parent()
            .ok_or_else(|| CortexError::SandboxViolation("Path has no parent directory".into()))?;

        let canonical_parent = std::fs::canonicalize(parent).map_err(|e| {
            CortexError::SandboxViolation(format!(
                "Cannot resolve parent directory of '{}': {}",
                path, e
            ))
        })?;

        let is_within_workspace = self
            .allowed_roots
            .iter()
            .any(|root| canonical_parent.starts_with(root));

        if !is_within_workspace {
            return Err(CortexError::SandboxViolation(format!(
                "Write path '{}' parent resolves to '{}' which is outside all allowed workspace roots",
                path,
                canonical_parent.display()
            )));
        }

        // Return the resolved (not necessarily canonical) full path,
        // since the file itself may not exist yet.
        let file_name = resolved.file_name().ok_or_else(|| {
            CortexError::SandboxViolation("Path has no filename component".into())
        })?;

        Ok(canonical_parent.join(file_name))
    }

    /// Validate and parse a shell command string.
    ///
    /// Returns the binary name and arguments as separate components.
    /// Rejects:
    /// - Empty commands
    /// - Binaries not in the allowlist
    /// - Unparseable command strings
    pub fn validate_command(&self, cmd: &str) -> Result<(String, Vec<String>), CortexError> {
        let parts: Vec<String> = shell_split(cmd)?;

        if parts.is_empty() {
            return Err(CortexError::SandboxViolation("Empty command string".into()));
        }

        let binary = &parts[0];

        // Extract the basename in case a path was provided (e.g., "/usr/bin/cargo" → "cargo")
        let binary_name = Path::new(binary)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(binary);

        if !self.allowed_commands.contains(binary_name) {
            return Err(CortexError::SandboxViolation(format!(
                "Command '{}' is not in the allowed commands list. Allowed: {:?}",
                binary_name, self.allowed_commands
            )));
        }

        let args = parts[1..].to_vec();
        Ok((binary_name.to_string(), args))
    }

    /// Get the canonicalized workspace roots.
    pub fn roots(&self) -> &[PathBuf] {
        &self.allowed_roots
    }

    /// Get the configured timeout duration.
    pub fn timeout(&self) -> Duration {
        self.max_timeout
    }
}

/// Simple POSIX-style shell argument splitting.
///
/// Handles single and double quotes. Does NOT handle:
/// - Backslash escapes outside quotes
/// - Shell expansions ($VAR, $(cmd), etc.)
/// - Redirections or pipes
///
/// This is intentional — we want structured command execution,
/// not a full shell interpreter.
fn shell_split(input: &str) -> Result<Vec<String>, CortexError> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escape_next = false;

    for ch in input.chars() {
        if escape_next {
            current.push(ch);
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if !in_single_quote => {
                escape_next = true;
            }
            '\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
            }
            '"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
            }
            ' ' | '\t' if !in_single_quote && !in_double_quote => {
                if !current.is_empty() {
                    parts.push(std::mem::take(&mut current));
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if in_single_quote || in_double_quote {
        return Err(CortexError::SandboxViolation(
            "Unterminated quote in command string".into(),
        ));
    }

    if !current.is_empty() {
        parts.push(current);
    }

    Ok(parts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_policy() -> SandboxPolicy {
        let tmp = std::env::temp_dir();
        let root = tmp.join("sandbox_test_workspace");
        let _ = fs::create_dir_all(&root);
        // Create a test file inside the workspace
        let _ = fs::write(root.join("allowed.txt"), "test");

        let canonical_root = fs::canonicalize(&root).unwrap();

        SandboxPolicy {
            allowed_roots: vec![canonical_root],
            allowed_commands: ["cargo", "git", "ls"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            clear_env: true,
            max_timeout: Duration::from_secs(10),
            wasm_fuel: 1_000_000,
            wasm_memory_pages: 256,
        }
    }

    #[test]
    fn test_validate_path_allows_workspace() {
        let policy = test_policy();
        let root = &policy.allowed_roots[0];
        let test_file = root.join("allowed.txt");
        let result = policy.validate_path(test_file.to_str().unwrap());
        assert!(
            result.is_ok(),
            "Should allow path inside workspace: {:?}",
            result
        );
    }

    #[test]
    fn test_validate_path_rejects_traversal() {
        let policy = test_policy();
        let result = policy.validate_path("../../etc/passwd");
        // This should either fail because /etc/passwd is outside workspace,
        // or fail because the resolved path doesn't exist
        assert!(result.is_err(), "Should reject path traversal");
    }

    #[test]
    fn test_validate_path_rejects_absolute_outside() {
        let policy = test_policy();
        let result = policy.validate_path("/etc/hosts");
        // /etc/hosts exists on macOS but is outside workspace
        assert!(
            result.is_err(),
            "Should reject absolute path outside workspace"
        );
    }

    #[test]
    fn test_validate_path_rejects_symlink_escape() {
        let policy = test_policy();
        let root = &policy.allowed_roots[0];

        // Create a symlink inside workspace pointing outside
        let link_path = root.join("escape_link");
        let _ = fs::remove_file(&link_path); // clean up any previous run
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink("/etc", &link_path);
            let result = policy.validate_path(link_path.join("hosts").to_str().unwrap());
            assert!(
                result.is_err(),
                "Should reject symlink that escapes workspace"
            );
            let _ = fs::remove_file(&link_path);
        }
    }

    #[test]
    fn test_validate_command_allows_listed() {
        let policy = test_policy();
        let result = policy.validate_command("cargo check --workspace");
        assert!(result.is_ok());
        let (binary, args) = result.unwrap();
        assert_eq!(binary, "cargo");
        assert_eq!(args, vec!["check", "--workspace"]);
    }

    #[test]
    fn test_validate_command_rejects_unlisted() {
        let policy = test_policy();
        let result = policy.validate_command("rm -rf /");
        assert!(result.is_err(), "Should reject commands not in allowlist");
    }

    #[test]
    fn test_validate_command_rejects_sh_c() {
        let policy = test_policy();
        // 'sh' is not in our allowlist
        let result = policy.validate_command("sh -c 'echo pwned'");
        assert!(result.is_err(), "Should reject sh -c");
    }

    #[test]
    fn test_validate_command_handles_quotes() {
        let policy = test_policy();
        let result = policy.validate_command(r#"cargo check --message-format="json""#);
        assert!(result.is_ok());
        let (binary, args) = result.unwrap();
        assert_eq!(binary, "cargo");
        assert_eq!(args, vec!["check", "--message-format=json"]);
    }

    #[test]
    fn test_shell_split_basic() {
        let parts = shell_split("hello world foo").unwrap();
        assert_eq!(parts, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_shell_split_quoted() {
        let parts = shell_split(r#"echo "hello world" foo"#).unwrap();
        assert_eq!(parts, vec!["echo", "hello world", "foo"]);
    }

    #[test]
    fn test_shell_split_single_quoted() {
        let parts = shell_split("echo 'hello world' foo").unwrap();
        assert_eq!(parts, vec!["echo", "hello world", "foo"]);
    }

    #[test]
    fn test_shell_split_unterminated() {
        let result = shell_split("echo 'hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_roots_denies_all() {
        let policy = SandboxPolicy {
            allowed_roots: vec![],
            allowed_commands: HashSet::new(),
            clear_env: true,
            max_timeout: Duration::from_secs(10),
            wasm_fuel: 0,
            wasm_memory_pages: 0,
        };
        let result = policy.validate_path("/any/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_dev_default_creates_valid_policy() {
        let policy = SandboxPolicy::dev_default();
        assert!(!policy.allowed_roots.is_empty());
        assert!(policy.allowed_commands.contains("cargo"));
        assert!(policy.allowed_commands.contains("git"));
        assert_eq!(policy.max_timeout, Duration::from_secs(30));
    }
}
