#!/usr/bin/env bash
# Re-apply pi-subagents acceptance patches after npm install.
#
# These patches fix two upstream bugs that cause acceptance-rejected runs:
#
#   F2: Parser only recognized ```acceptance-report fences, rejecting
#       the much more common ```json fences that models actually emit.
#       Models frequently write valid JSON but with a non-standard fence
#       tag. The new parser is lenient: it tries ```acceptance-report,
#       then ```json/jsonc/json5, then ACCEPTANCE_REPORT: marker, then
#       any standalone balanced JSON object that matches the
#       AcceptanceReport shape. Also tolerates trailing commas, JSONC
#       comments, leading prose inside the fence, and `{acceptance: {...}}`
#       wrappers.
#
#   F9: Prompt didn't clearly tell the model what fence tag to use.
#       Strengthened both the initial contract prompt and the
#       finalization prompt to explicitly say:
#         - Use exactly `acceptance-report` (not `json`)
#         - Don't write prose without JSON
#         - Don't say "Let me now write..." and stop
#
#   F10: getFinalOutput() returned only the LAST assistant text message,
#        so if the model emitted the JSON one turn earlier than its
#        final "Done." text, the JSON was silently discarded. Fix:
#        prefer messages containing an acceptance-report block when
#        extracting final output.
#
# Apply order matters: 01 → 02 → 03 → 04. Re-running is idempotent if
# patch is already applied (patch -p1 --forward handles this).
#
# Usage:
#   ./apply-patches.sh                # apply to the npm-installed copy
#   ./apply-patches.sh --check        # check whether patches are applied
#
# After applying, RESTART pi (the parent process caches modules in memory).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${PI_SUBAGENTS_TARGET:-/home/tsuser/.pi/agent/npm/node_modules/pi-subagents}"
CHECK_ONLY=0

if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "✗ Target directory not found: $TARGET_DIR"
  exit 1
fi

cd "$TARGET_DIR"
echo "Target: $TARGET_DIR"
echo

PATCHES=(
  "$SCRIPT_DIR/01-lenient-acceptance-parser.patch"
  "$SCRIPT_DIR/02-stronger-acceptance-prompt.patch"
  "$SCRIPT_DIR/03-stronger-finalization-prompt.patch"
  "$SCRIPT_DIR/04-prefer-acceptance-message.patch"
)

applied=0
skipped=0
failed=0
for patch in "${PATCHES[@]}"; do
  name=$(basename "$patch")
  if [[ $CHECK_ONLY -eq 1 ]]; then
    if patch -p1 --dry-run --forward --silent < "$patch" 2>/dev/null; then
      echo "  NEEDS APPLY: $name"
      applied=$((applied + 1))
    else
      echo "  already applied: $name"
      skipped=$((skipped + 1))
    fi
    continue
  fi
  if patch -p1 --forward --silent < "$patch" 2>/dev/null; then
    echo "  ✓ applied: $name"
    applied=$((applied + 1))
  else
    # patch returns non-zero if already applied (with --forward)
    if patch -p1 --dry-run --reverse --silent < "$patch" 2>/dev/null; then
      echo "  already applied: $name"
      skipped=$((skipped + 1))
    else
      echo "  ✗ FAILED: $name"
      failed=$((failed + 1))
    fi
  fi
done

echo
if [[ $CHECK_ONLY -eq 1 ]]; then
  if [[ $applied -eq 0 ]]; then
    echo "✓ All patches applied."
    exit 0
  else
    echo "✗ $applied patch(es) need to be applied. Run $0 without --check."
    exit 1
  fi
fi

if [[ $failed -gt 0 ]]; then
  echo "✗ $failed patch(es) failed."
  exit 1
fi

if [[ $skipped -gt 0 && $applied -eq 0 ]]; then
  echo "✓ All patches were already applied."
  exit 0
fi

cat <<EOF

✓ Applied $applied new patch(es). $skipped were already applied.

⚠ IMPORTANT: pi caches modules in memory. Restart pi to pick up the fix:
   pkill -x pi   # or restart your terminal
EOF
