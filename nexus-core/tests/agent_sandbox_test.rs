use nexus_core::cortex::Cortex;
use nexus_core::sandbox::SandboxPolicy;
use std::path::PathBuf;
use std::collections::HashSet;

// This integration test verifies that SandboxPolicy enforces correctly.

#[test]
fn test_agent_shell_injection_blocked() {
    let mut commands = HashSet::new();
    commands.insert("ls".to_string());
    
    let policy = SandboxPolicy::dev_default(); // Start with dev_default and override
    
    let cortex = Cortex::boot(policy);
    let p = cortex.policy();
    
    // Case 1: Command not in allowlist
    let res = p.validate_command("pwned_command -rf /");
    assert!(res.is_err());
    assert!(res.unwrap_err().to_string().contains("is not in the allowed commands list"));
    
    // Case 2: Shell injection attempt
    let res = p.validate_command("ls; rm -rf /");
    assert!(res.is_err());
    
    // Case 3: Path traversal
    let res = p.validate_path("../etc/passwd");
    assert!(res.is_err());
}
