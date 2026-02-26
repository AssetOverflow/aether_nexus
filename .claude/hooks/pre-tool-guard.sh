#!/usr/bin/env bash
# AetherNexus pre-tool guardrail hook
# Called by Claude Code before every Bash tool invocation.
# Exit 2 = hard block (tool is denied, message shown to user).
# Exit 0 = allow.
#
# Claude Code passes the tool input as JSON on stdin.
# We extract the command string and pattern-match against a deny list.

set -euo pipefail

input=$(cat)
command=$(echo "$input" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('command',''))" 2>/dev/null || echo "")

deny() {
  echo "$1" >&2
  exit 2
}

# ── Destructive git operations ────────────────────────────────────────────────

if echo "$command" | grep -qE 'git\s+push\s+.*--force|git\s+push\s+.*-f\b'; then
  deny "BLOCKED: Force-push is permanently denied. Rewrite history locally if needed, then ask the user to push manually."
fi

if echo "$command" | grep -qE 'git\s+reset\s+--hard'; then
  deny "BLOCKED: 'git reset --hard' is permanently denied — it discards uncommitted work. Use 'git stash' or commit first."
fi

if echo "$command" | grep -qE 'git\s+clean\s+.*-f'; then
  deny "BLOCKED: 'git clean -f' is permanently denied — it permanently deletes untracked files."
fi

if echo "$command" | grep -qE 'git\s+branch\s+.*-D\b'; then
  deny "BLOCKED: Force-deleting branches ('git branch -D') is permanently denied. Use '-d' (safe delete) and ask the user for force deletes."
fi

# ── Destructive file operations ───────────────────────────────────────────────

if echo "$command" | grep -qE 'rm\s+.*-[a-zA-Z]*r[a-zA-Z]*f|rm\s+.*-[a-zA-Z]*f[a-zA-Z]*r'; then
  deny "BLOCKED: 'rm -rf' is permanently denied. Use targeted file removal and confirm with the user."
fi

# Protect .aether files — the mmap state is irreplaceable without genesis
if echo "$command" | grep -qE 'rm\s+.*\.aether|mv\s+.*\.aether|truncate\s+.*\.aether|dd\s+.*\.aether'; then
  deny "BLOCKED: Direct removal or overwrite of .aether files is permanently denied. These files contain the signed Fabric state. Perform genesis explicitly if you intend to reset."
fi

# Protect critical config and key material
if echo "$command" | grep -qE '(>|>>|tee)\s*(~\/\.ssh|\/etc\/|\/usr\/|\/System\/|~\/\.gnupg)'; then
  deny "BLOCKED: Writing to system directories (~/.ssh, /etc, /usr, /System, ~/.gnupg) is permanently denied."
fi

# ── Network calls (air-gapped runtime principle) ──────────────────────────────

if echo "$command" | grep -qE '\bcurl\b|\bwget\b|\bfetch\b'; then
  deny "BLOCKED: Network fetch commands (curl, wget) are permanently denied in this project. AetherNexus is air-gapped — all assets must already be present locally."
fi

# ── Cargo publish / release ───────────────────────────────────────────────────

if echo "$command" | grep -qE 'cargo\s+publish'; then
  deny "BLOCKED: 'cargo publish' is permanently denied. Releases must be performed manually by the user."
fi

exit 0
