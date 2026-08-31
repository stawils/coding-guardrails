#!/usr/bin/env python3
"""Live smoke test for a RUNNING coding-guardrails proxy (:8081).

Codifies the 2026-08-31 reliability-session live checks so they are
repeatable in one command:

- /health responds ok (and carries backend state when managed)
- /v1/models lists the served model
- tool-path responses carry thinking (reasoning_content, keep-last default)
- trailing-system-message history is accepted (role normalization; was 400)
- multi-turn history with STRING tool-call args is accepted (wire coercion; was 400)
- unversioned POST /chat/completions alias works (forge 0.8.2 compat)
- no-tool requests return a clean direct answer (auto_no_thinking)

Usage: python scripts/smoke_proxy.py [base_url]   (default http://localhost:8081)
Exit 0 = all green; non-zero = failures listed.
"""

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8081"
MODEL = "Qwen3.8-27B-UD-Q3_K_XL"
BASH = {"type": "function", "function": {
    "name": "bash", "description": "Run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]},
}}
failures: list[str] = []


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> None:
    # 1. health
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        health = json.loads(r.read())
    check("health ok", health.get("status") == "ok", json.dumps(health)[:120])

    # 2. models
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=10) as r:
        models = json.loads(r.read())
    check("models listed", any("Qwen3.8" in m.get("id", "") for m in models.get("data", [])),
          json.dumps(models.get("data", []))[:100])

    # 3. tool path carries thinking
    d = post("/v1/chat/completions", {"model": MODEL, "stream": False, "tools": [BASH],
        "messages": [{"role": "user",
                      "content": "Use the bash tool with command 'echo smoke-1' and report the output."}]})
    msg = d.get("choices", [{}])[0].get("message", {})
    check("tool path reasoning_content", "reasoning_content" in msg, f"tools={len(msg.get('tool_calls') or [])}")

    # 4. trailing system accepted (role normalization)
    d = post("/v1/chat/completions", {"model": MODEL, "stream": False, "tools": [BASH],
        "messages": [{"role": "user", "content": "Run bash 'echo smoke-2'."},
                     {"role": "system", "content": "trailing project instructions"}]})
    check("trailing system accepted", "error" not in d)

    # 5. multi-turn string-args history accepted (wire coercion)
    d = post("/v1/chat/completions", {"model": MODEL, "stream": False, "tools": [BASH],
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "bash", "arguments": "{\"command\": \"echo old\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "old"},
            {"role": "user", "content": "Run bash 'echo smoke-3'."}]})
    check("string-args history accepted", "error" not in d)

    # 6. unversioned alias
    d = post("/chat/completions", {"model": MODEL, "stream": False,
        "messages": [{"role": "user", "content": "reply with: alias-ok"}]})
    check("unversioned alias", "error" not in d,
          (d.get("choices", [{}])[0].get("message", {}).get("content") or "")[:30])

    # 7. no-tool direct answer
    d = post("/v1/chat/completions", {"model": MODEL, "stream": False,
        "messages": [{"role": "user", "content": "Reply with exactly: direct"}]})
    msg = d.get("choices", [{}])[0].get("message", {})
    check("no-tool direct answer", (msg.get("content") or "").strip().startswith("direct"),
          repr((msg.get("content") or "")[:40]))

    print(f"\n{'ALL GREEN' if not failures else f'{len(failures)} FAILURES'}: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()