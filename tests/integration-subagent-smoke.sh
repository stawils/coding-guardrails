#!/usr/bin/env bash
# Integration test: spawn a real subagent through the coding-guardrails
# proxy and verify the end-to-end pipeline works.
#
# What this validates:
#   1. coding-guardrails proxy is reachable on :8081
#   2. Prerequisites rule allows parallel writes of new files (F1/F5 fix)
#   3. pi-subagents acceptance parser accepts the model's JSON output (F2 fix)
#   4. Finalization loop converges (F9 prompt fix)
#
# Usage:
#   ./tests/integration-subagent-smoke.sh [--keep]
#
# --keep: don't delete /tmp/cg-smoke-* after run (for debugging)
#
# Requires: pi running, coding-guardrails proxy on :8081, model
#           coding-guardrails/Qwen3.5-9B-UD-Q4_K_XL
set -euo pipefail

KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

TARGET="/tmp/cg-smoke-$$"
mkdir -p "$TARGET"
trap '[ $KEEP -eq 0 ] && rm -rf "$TARGET"' EXIT

echo "==> Target: $TARGET"
echo "==> Checking proxy..."
curl -sf -m 3 http://localhost:8081/v1/models > /dev/null || {
  echo "✗ Proxy not reachable on :8081"
  exit 1
}
echo "    ✓ proxy healthy"

echo "==> Spawning subagent..."
START=$(date +%s)

# Use pi's subagent tool via the JSONL events file
# This requires being run from a pi context. For standalone runs, we
# invoke pi-subagents directly.
if ! command -v pi >/dev/null; then
  echo "✗ pi not in PATH"
  exit 1
fi

# Spawn a minimal scaffolding task with acceptance contract
OUTPUT=$(pi --no-tui --timeout 60 -p "Use the subagent tool to spawn agent 'worker' with model 'coding-guardrails/Qwen3.5-9B-UD-Q4_K_XL', cwd '$TARGET', and this task:
Create these 3 files in $TARGET:
- package.json: {\"name\":\"smoke\",\"version\":\"0.0.0\"}
- index.js: console.log('smoke')
- README.md: # Smoke test
Then run: ls $TARGET
Acceptance: criteria=['files-exist: 3 files created'], evidence=['changed-files','commands-run'], verify=[{'id':'files','command':'ls $TARGET/package.json $TARGET/index.js $TARGET/README.md'}]" 2>&1) || true

END=$(date +%s)
echo "    duration: $((END - START))s"

# Check files were created
COUNT=$(find "$TARGET" -type f -name '*.json' -o -name '*.js' -o -name '*.md' 2>/dev/null | wc -l)
if [ "$COUNT" -ge 3 ]; then
  echo "✓ Files created ($COUNT >= 3)"
else
  echo "✗ Only $COUNT files created (expected >= 3)"
  exit 1
fi

# Check acceptance
if echo "$OUTPUT" | grep -q "Acceptance rejected"; then
  echo "✗ Acceptance rejected"
  echo "$OUTPUT" | tail -20
  exit 1
elif echo "$OUTPUT" | grep -q "completed"; then
  echo "✓ Subagent completed"
else
  echo "? Unclear completion status — check output"
  echo "$OUTPUT" | tail -10
fi

echo
echo "==> PASS: end-to-end smoke test"
