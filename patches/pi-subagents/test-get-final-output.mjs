// Test getFinalOutput prefers messages with acceptance-report blocks.
import { readFileSync } from "node:fs";

// Inline the function under test (since it's TypeScript with imports)
// We'll read the source file and eval the function definition.
const src = readFileSync("/home/tsuser/.pi/agent/npm/node_modules/pi-subagents/src/shared/utils.ts", "utf8");

// Quick-and-dirty: extract just the function and its dependencies
// The function doesn't depend on Message type at runtime (duck-typed)
const fnMatch = src.match(/export function getFinalOutput[\s\S]*?\n\}/);
if (!fnMatch) {
  console.error("Could not extract getFinalOutput");
  process.exit(2);
}
const fnSrc = fnMatch[0]
  .replace("export function getFinalOutput", "function getFinalOutput")
  .replace(/messages: Message\[\]\):\s*string\s*\{/, "messages) {");
const getFinalOutput = new Function(fnSrc + "\nreturn getFinalOutput;")();

const testCases = [
  {
    name: "Strategy 1: prefer message with acceptance-report block",
    messages: [
      { role: "user", content: [{ type: "text", text: "do work" }] },
      { role: "assistant", content: [{ type: "text", text: "Working on it..." }] },
      { role: "user", content: [{ type: "text", text: "show report" }] },
      { role: "assistant", content: [{ type: "text", text: "```acceptance-report\n{ ... }\n```" }] },
      { role: "user", content: [{ type: "text", text: "ok thanks" }] },
      { role: "assistant", content: [{ type: "text", text: "You're welcome." }] },
    ],
    expectContains: "acceptance-report",
  },
  {
    name: "Strategy 2: fall back to last text if no acceptance-report anywhere",
    messages: [
      { role: "user", content: [{ type: "text", text: "hi" }] },
      { role: "assistant", content: [{ type: "text", text: "All done." }] },
    ],
    expectContains: "All done.",
  },
  {
    name: "Skip error-flagged assistant messages",
    messages: [
      { role: "assistant", content: [{ type: "text", text: "good text" }] },
      { role: "assistant", content: [{ type: "text", text: "bad text" }], errorMessage: "model crashed" },
    ],
    expectContains: "good text",
  },
  {
    name: "ACCEPTANCE_REPORT: marker also matches",
    messages: [
      { role: "assistant", content: [{ type: "text", text: "Done." }] },
      { role: "assistant", content: [{ type: "text", text: "ACCEPTANCE_REPORT:\n{ \"x\": 1 }" }] },
      { role: "assistant", content: [{ type: "text", text: "Final text." }] },
    ],
    expectContains: "ACCEPTANCE_REPORT",
  },
  {
    name: "Empty messages → empty string",
    messages: [],
    expectContains: "",
  },
];

let pass = 0, fail = 0;
for (const c of testCases) {
  const result = getFinalOutput(c.messages);
  const ok = c.expectContains === "" ? result === "" : result.includes(c.expectContains);
  console.log(`[${ok ? "PASS" : "FAIL"}] ${c.name}`);
  console.log(`         got: ${JSON.stringify(result).slice(0, 100)}`);
  if (ok) pass++; else fail++;
}
console.log(`\n${pass} pass, ${fail} fail`);
process.exit(fail > 0 ? 1 : 0);
