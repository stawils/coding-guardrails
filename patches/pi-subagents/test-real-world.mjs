// Real-world test: feed actual failed outputs from the Lirada session
// through the new lenient parser.
import { readFileSync } from "node:fs";
import { parseAcceptanceReport } from "/home/tsuser/.pi/agent/npm/node_modules/pi-subagents/src/runs/shared/acceptance-reports.ts";

const cases = [
  { name: "c5ee1aaf (real, ```json fence with perfect JSON)", file: "/home/tsuser/.pi/agent/sessions/--home-tsuser-AI-libya-forex-tracker--/subagent-artifacts/c5ee1aaf_worker_0_output.md" },
  { name: "6b681ef4 (real, ```json fence + 'acceptance-report' label)", file: "/home/tsuser/.pi/agent/sessions/--home-tsuser-workspaces--/subagent-artifacts/6b681ef4_worker_0_output.md" },
  { name: "84754494 (real, model stopped before emitting)", file: "/home/tsuser/.pi/agent/sessions/--home-tsuser-AI-libya-forex-tracker--/subagent-artifacts/84754494_worker_0_output.md" },
  { name: "b75ec4d7 (real, prose only)", file: "/home/tsuser/.pi/agent/sessions/--home-tsuser-AI-libya-forex-tracker--/subagent-artifacts/b75ec4d7_worker_0_output.md" },
  { name: "1a8f44dd (real, prose only)", file: "/home/tsuser/.pi/agent/sessions/--home-tsuser-workspaces--/subagent-artifacts/1a8f44dd_worker_0_output.md" },
];

for (const c of cases) {
  const input = readFileSync(c.file, "utf8");
  const result = parseAcceptanceReport(input);
  const status = result.report ? "✓ PARSED" : "✗ MISSED";
  console.log(`${status}  ${c.name}`);
  if (result.report) {
    console.log(`         changedFiles: ${result.report.changedFiles?.length ?? 0}`);
    console.log(`         commandsRun: ${result.report.commandsRun?.length ?? 0}`);
    console.log(`         criteria: ${result.report.criteriaSatisfied?.length ?? 0}`);
  } else {
    console.log(`         error: ${result.error}`);
  }
}
