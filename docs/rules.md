# Rules Reference

coding-guardrails ships with 6 guardrail rules, each independently configurable.

## Rule Behavior

- **Hard** — blocks the tool call and returns an error to the agent
- **Soft** — allows the call through but adds a nudge/advisory message
- All rules can be **disabled** individually via config

---

## 1. Path Safety (`path_safety`)

**What it does:** Blocks file operations outside the allowed workspace.

**Default:** Hard block.

| Setting | Default | Description |
|---|---|---|
| `blocked_prefixes` | `/etc/`, `/sys/`, `/proc/`, etc. | Always-blocked paths |
| `allowed_prefixes` | (none) | Additional allowed paths outside workspace |
| `workspace` | `.` | Workspace root directory |

**Examples:**
- ❌ `read_file("/etc/passwd")` → blocked: outside workspace
- ❌ `write_file("/root/.ssh/authorized_keys", ...)` → blocked
- ✅ `read_file("src/main.py")` → allowed: inside workspace

---

## 2. Command Safety (`command_safety`)

**What it does:** Blocks destructive shell commands.

**Default:** Hard block.

| Setting | Default | Description |
|---|---|---|
| `blocked_commands` | `rm -rf /`, `mkfs`, `dd`, etc. | Blocked command patterns |
| `strength` | `hard` | `hard` = block, `soft` = warn only |

**Examples:**
- ❌ `bash("rm -rf /")` → blocked: destructive command
- ❌ `bash("mkfs.ext4 /dev/sda1")` → blocked: disk formatting
- ❌ `bash(":(){ :|:& };:")` → blocked: fork bomb
- ✅ `bash("npm install")` → allowed

---

## 3. Secret Detection (`secrets`)

**What it does:** Detects and masks secrets in tool call arguments.

**Default:** Hard block (prevents secrets from being passed through).

| Setting | Default | Description |
|---|---|---|
| `strength` | `hard` | `hard` = block, `soft` = mask and warn |
| `mask_value` | `[REDACTED]` | Replacement for detected secrets |

**Detects:**
- AWS Access Key IDs (`AKIA...`)
- AWS Secret Access Keys
- GitHub tokens (`ghp_`, `gho_`, `ghu_`, `ghs_`)
- Generic API keys (40+ char hex/base64)
- RSA/EC private keys (`-----BEGIN ... PRIVATE KEY-----`)
- Bearer tokens

**Examples:**
- ❌ `bash("export AWS_SECRET_ACCESS_KEY=wJalr...")` → blocked: secret detected
- ❌ `write_file("~/.ssh/id_rsa", "-----BEGIN RSA PRIVATE KEY-----")` → blocked
- ✅ `bash("export PATH=/usr/bin")` → allowed: no secrets

---

## 4. Prerequisites (`prerequisites`)

**What it does:** Ensures files are read before editing.

**Default:** Soft nudge (suggests reading first).

| Setting | Default | Description |
|---|---|---|
| `strength` | `soft` | `soft` = suggest, `hard` = block until read |
| `cooldown` | `3` | Turns before re-nudging |
| `edit_tools` | `write_file`, `edit_file`, `create_file` | Tools that modify files |
| `read_tools` | `read_file`, `cat` | Tools that read files |

**Examples:**
- ⚠️ `edit_file("config.yaml", ...)` (without prior read) → nudge: "read the file first"
- ✅ `read_file("config.yaml")` then `edit_file("config.yaml", ...)` → allowed

---

## 5. Sequencing (`sequencing`)

**What it does:** Suggests running tests after code changes.

**Default:** Soft nudge.

| Setting | Default | Description |
|---|---|---|
| `strength` | `soft` | `soft` = suggest, `hard` = block until test |
| `cooldown` | `3` | Turns before re-suggesting |
| `test_commands` | `pytest`, `cargo test`, `go test` | Commands recognized as tests |

**Examples:**
- ⚠️ `edit_file("main.py", ...)` (without subsequent test) → nudge: "run tests"
- ✅ `edit_file("main.py", ...)` then `bash("pytest")` → no nudge

---

## 6. Tool Resolution (`tool_resolution`)

**What it does:** Warns when tool results are empty, whitespace-only, or error messages.

**Default:** Soft nudge.

| Setting | Default | Description |
|---|---|---|
| `strength` | `soft` | Always soft — informational only |

**Detects:**
- Empty tool results
- Whitespace-only results
- Error messages (`Error:`, `Permission denied`, `No such file`)

**Examples:**
- ⚠️ Tool returns `""` → nudge: "tool returned empty result"
- ⚠️ Tool returns `"Permission denied"` → nudge: check permissions
- ✅ Tool returns file contents → no nudge
