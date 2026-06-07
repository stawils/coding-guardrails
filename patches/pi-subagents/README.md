# pi-subagents acceptance patches

> **STATUS (2026-06-07): NOT APPLIED.** These patches are kept on disk as
> the documented basis for upstream issue
> [nicobailon/pi-subagents#253](https://github.com/nicobailon/pi-subagents/issues/253)
> only. They are **no longer applied** to the installed pi-subagents — the
> running pi uses the **pristine v0.28.0 tarball** (verified byte-identical).
> Reliability is instead handled at the orchestration layer: delegations run
> without `acceptance` contracts, and verification (build/test) is run
> directly by the orchestrator. Do **not** re-apply these patches without an
> explicit decision; that would reintroduce local modifications to pi's
> source. `apply-patches.sh --check` will report all 5 as "NEEDS APPLY" —
> that is the expected/desired state.

Fixes for upstream pi-subagents 0.28.0 that cause valid acceptance reports
to be silently rejected. Discovered during the 2026-06-05 Lirada build
session where ~60% of subagent runs failed acceptance despite the model
emitting valid JSON reports.

## Bugs fixed

| ID | Symptom | Root cause |
|----|---------|-----------|
| **F2** | "Structured acceptance report not found" even though the model emitted perfect JSON | Parser only recognized `` ```acceptance-report `` fence tag, rejected the much more common `` ```json `` tag that models actually emit |
| **F9** | Model writes prose like "Let me now write the report..." and stops | Prompt didn't clearly forbid this; new prompt explicitly says "use `acceptance-report` tag, not `json`" and lists wrong patterns |
| **F10** | Report from turn N-1 silently discarded because turn N was a summary | `getFinalOutput()` returned only the LAST assistant text. Now prefers messages containing an `acceptance-report` block |

## Files modified

| Patch | File | Change |
|-------|------|--------|
| `01-lenient-acceptance-parser.patch` | `src/runs/shared/acceptance-reports.ts` | Multi-strategy parser: tries `acceptance-report` → `json`/`jsonc` → `ACCEPTANCE_REPORT:` marker → raw JSON. Tolerates trailing commas, JSONC comments, leading prose, `{acceptance: {...}}` wrappers. |
| `02-stronger-acceptance-prompt.patch` | `src/runs/shared/acceptance-contract.ts` | Initial contract prompt now explicitly states the fence tag and lists wrong patterns. |
| `03-stronger-finalization-prompt.patch` | `src/runs/shared/acceptance-finalization.ts` | Finalization prompt now says "EXACTLY ONE block, tagged `acceptance-report` (not `json`)". |
| `04-prefer-acceptance-message.patch` | `src/shared/utils.ts` | `getFinalOutput()` strategy 1: prefer message containing acceptance-report block. Strategy 2: original "last text" behavior. |

## Apply

```bash
./apply-patches.sh             # apply to npm-installed copy
./apply-patches.sh --check     # check whether patches are needed
```

Defaults target to `/home/tsuser/.pi/agent/npm/node_modules/pi-subagents`.
Override with `PI_SUBAGENTS_TARGET=/path/to/pi-subagents ./apply-patches.sh`.

**After applying, restart pi** — the parent process caches TypeScript
modules in memory. `pkill -x pi` or restart your terminal.

## Verification

Tested against real failed runs from the Lirada session:

- **c5ee1aaf**: previously rejected (used `` ```json `` fence). Now parses → 10 changedFiles, 3 commandsRun, 2 criteria satisfied.
- **6b681ef4**: previously rejected (used `` ```json `` + "acceptance-report" label inside). Now parses → 1 changedFile, 4 commandsRun, 4 criteria satisfied.
- **bf30a057 msg 22**: produced during finalization loop but silently discarded by old `getFinalOutput`. Now both parses AND is preferred by the new "strategy 1" extraction.

## Tests

```bash
# Parser unit tests (10 cases — happy paths + negatives)
bun /tmp/test-acceptance-parser.mjs

# getFinalOutput tests (5 cases)
bun /tmp/test-get-final-output.mjs

# Real-world data test
bun /tmp/test-real-world.mjs
```

## Upstreaming

These patches should land in pi-subagents upstream. The parser change is
backwards-compatible (still accepts all old formats). The prompt changes
are additive. The `getFinalOutput` change is the only behavior change —
it's strictly more permissive (will accept reports that were previously
lost).

Repository: https://github.com/nicobailon/pi-subagents
