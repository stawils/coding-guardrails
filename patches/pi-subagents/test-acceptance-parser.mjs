import { parseAcceptanceReport } from "/home/tsuser/.pi/agent/npm/node_modules/pi-subagents/src/runs/shared/acceptance-reports.ts";

const cases = [
  {
    name: "F2a: ```json fence with 'acceptance-report' label inside",
    input: `Done!

\`\`\`json
acceptance-report
{
  "criteriaSatisfied": [
    { "id": "criterion-1", "status": "satisfied", "evidence": "files created" }
  ],
  "changedFiles": ["/tmp/foo.js"],
  "commandsRun": [],
  "validationOutput": [],
  "residualRisks": [],
  "diffSummary": "added foo"
}
\`\`\``,
  },
  {
    name: "F2a-clean: ```json fence with raw JSON (no label)",
    input: `Done!

\`\`\`json
{
  "criteriaSatisfied": [
    { "id": "criterion-1", "status": "satisfied", "evidence": "ok" }
  ],
  "changedFiles": ["/tmp/bar.js"],
  "diffSummary": "added bar"
}
\`\`\``,
  },
  {
    name: "F2c: Classic ```acceptance-report fence (must still work)",
    input: `\`\`\`acceptance-report
{
  "criteriaSatisfied": [
    { "id": "c1", "status": "satisfied", "evidence": "yes" }
  ],
  "changedFiles": []
}
\`\`\``,
  },
  {
    name: "F2d: ACCEPTANCE_REPORT: marker",
    input: `Done.

ACCEPTANCE_REPORT:
{
  "criteriaSatisfied": [
    { "id": "x", "status": "satisfied", "evidence": "y" }
  ],
  "changedFiles": [],
  "diffSummary": "done"
}`,
  },
  {
    name: "F2e: Raw JSON, no fence at all",
    input: `Here is my report:

{
  "criteriaSatisfied": [
    { "id": "raw", "status": "satisfied", "evidence": "raw json" }
  ],
  "changedFiles": ["x"],
  "diffSummary": "no fence"
}`,
  },
  {
    name: "F2f: Wrapped in { acceptance: {...} }",
    input: `\`\`\`json
{
  "acceptance": {
    "criteriaSatisfied": [
      { "id": "wrap", "status": "satisfied", "evidence": "wrapped" }
    ],
    "changedFiles": []
  }
}
\`\`\``,
  },
  {
    name: "F2g: JSON with trailing commas (common model error)",
    input: `\`\`\`json
{
  "criteriaSatisfied": [
    { "id": "tc", "status": "satisfied", "evidence": "trailing commas", }
  ],
  "changedFiles": ["a", "b",],
  "diffSummary": "tc",
}
\`\`\``,
  },
  {
    name: "Negative: unrelated json fence (should fail)",
    input: `\`\`\`json
{ "name": "vite", "version": "6.0.0" }
\`\`\``,
  },
  {
    name: "Negative: empty output (should fail)",
    input: ``,
  },
  {
    name: "Negative: prose only, no JSON (should fail)",
    input: `All done. Build passed. No JSON to see here.`,
  },
];

let pass = 0, fail = 0;
for (const c of cases) {
  const result = parseAcceptanceReport(c.input);
  const isNegative = c.name.startsWith("Negative");
  const success = isNegative ? !result.report : Boolean(result.report);
  const status = success ? "PASS" : "FAIL";
  if (success) pass++; else fail++;
  console.log(`[${status}] ${c.name}`);
  if (!success) {
    console.log(`  → ${result.error ?? "got report but expected failure"}`);
    if (result.report) console.log(`  → report: ${JSON.stringify(result.report).slice(0, 200)}`);
  }
}

console.log(`\n${pass} pass, ${fail} fail`);
process.exit(fail > 0 ? 1 : 0);
