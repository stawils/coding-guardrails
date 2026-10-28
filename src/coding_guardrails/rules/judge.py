"""Semantic judge client — calibrated probability decider over HTTP.

This is the *escape hatch* behind the deterministic rules, never a replacement
for them. A rule consults the judge only after its own patterns found nothing;
the judge answers typed questions with calibrated probabilities instead of
generated text, and the caller maps them onto a grade (allow / nudge / block).

Transport is HTTP to a separate service (`dlserver`, default :8044). No model
is ever loaded in the proxy process: no torch, no VRAM, and if the judge is
down or slow the request path is unaffected (fail-open, logged).

Design notes:
- Fail-open on every failure mode (timeout, connection error, non-200,
  malformed JSON). An unavailable judge must never block work.
- No retries. The configured timeout is the whole budget; a retry doubles it.
- Default off: without a ``judge:`` config block nothing here is constructed.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("coding_guardrails.layer2.judge")

# Outcome labels recorded on RuleResult.judge_outcome for audit.
OUTCOME_BLOCK = "block"
OUTCOME_NUDGE = "nudge"
OUTCOME_ALLOW = "allow"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_ERROR = "error"


@dataclass(frozen=True)
class JudgeDecision:
    """One judge answer.

    Attributes:
        p: Highest probability across the configured probes (0.0 on failure).
        answers: Per-probe probabilities, keyed by question name.
        raw: Raw response payload, for logging/inspection only.
        error: Failure label ("timeout" | "error" | "disabled"), else None.
        engine: Engine that answered (or the configured one on failure).
        elapsed_ms: Wall-clock time spent, including connection setup.
        gating: Probe names that may drive a verdict (noul-style answers).
            Choice/score probes are informational and never gate.
    """

    p: float = 0.0
    answers: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] | None = None
    error: str | None = None
    engine: str | None = None
    elapsed_ms: float = 0.0
    gating: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.error is not None

    def probe(self, name: str) -> float:
        """Probability for one probe, 0.0 when missing."""
        return float(self.answers.get(name, 0.0))

    def top_probe(self) -> str | None:
        """Name of the highest-probability gating probe, or None."""
        names = [k for k in self.answers if not self.gating or k in self.gating]
        if not names:
            return None
        return max(names, key=lambda k: self.answers[k])


class DecisionClient(Protocol):
    """Protocol for judge clients (real HTTP or a test fake)."""

    @property
    def engine(self) -> str: ...

    def decide(self, state: str, questions: dict[str, dict]) -> JudgeDecision: ...


class NullDecisionClient:
    """Always-allow client used when the judge is disabled or misconfigured."""

    def __init__(self, engine: str = "disabled") -> None:
        self._engine = engine

    @property
    def engine(self) -> str:
        return self._engine

    def decide(self, state: str, questions: dict[str, dict]) -> JudgeDecision:
        return JudgeDecision(p=0.0, error="disabled", engine=self._engine)


def _extract_probability(answer: Any) -> float | None:
    """Pull a probability out of one answer object.

    Accepts the decision layer's shapes:
      noul:  {"type": "noul", "noul": 0.93, "probabilities": {...}}
      choice/score: {"type": "choice", "choice": "x", "probabilities": {"x": 0.9, ...}}
    """
    if isinstance(answer, (int, float)):
        return float(answer)
    if not isinstance(answer, dict):
        return None

    for key in ("noul", "probability", "p", "confidence"):
        value = answer.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        # Prefer an explicit "true"/"yes" bucket for noul-style questions;
        # otherwise the highest-probability option is the probe's answer.
        for key in ("true", "yes", "True", "Yes"):
            if isinstance(probs.get(key), (int, float)):
                return float(probs[key])
        numeric = [v for v in probs.values() if isinstance(v, (int, float))]
        if numeric:
            return float(max(numeric))

    score = answer.get("score")
    if isinstance(score, (int, float)):
        return float(score)

    return None


class HTTPDecisionClient:
    """POST {url}/v1/decide and normalise the answers."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8044",
        engine: str = "decider-0.8b",
        timeout_ms: int = 500,
    ) -> None:
        self._url = url.rstrip("/")
        self._engine = engine
        self._timeout_s = max(timeout_ms, 1) / 1000.0

    @property
    def engine(self) -> str:
        return self._engine

    def decide(self, state: str, questions: dict[str, dict]) -> JudgeDecision:
        body = {
            "state": state,
            "questions": questions,
            "options": {"engine": self._engine, "echo": True},
        }
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self._url}/v1/decide",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                if response.status != 200:
                    return self._failure(f"error:http-{response.status}", started)
                raw = json.loads(response.read().decode("utf-8", errors="replace"))
        except (socket.timeout, TimeoutError):
            return self._failure(OUTCOME_TIMEOUT, started)
        except urllib.error.HTTPError as exc:
            return self._failure(f"error:http-{exc.code}", started)
        except (urllib.error.URLError, OSError) as exc:
            # URLError wraps socket timeouts on some paths.
            label = OUTCOME_TIMEOUT if "timed out" in str(exc).lower() else OUTCOME_ERROR
            return self._failure(label, started)
        except (ValueError, TypeError, UnicodeDecodeError):
            return self._failure(OUTCOME_ERROR, started)

        elapsed = (time.monotonic() - started) * 1000.0
        answers_raw = raw.get("answers") if isinstance(raw, dict) else None
        if not isinstance(answers_raw, dict):
            return self._failure(OUTCOME_ERROR, started, raw=raw if isinstance(raw, dict) else None)

        probabilities: dict[str, float] = {}
        noul: list[float] = []
        noul_names: set[str] = set()
        for name in questions:
            answer = answers_raw.get(name)
            p = _extract_probability(answer)
            if p is None:
                continue
            probabilities[name] = max(0.0, min(1.0, p))
            # Only probability-of-true answers drive the gate. A choice
            # question's winning option can be 1.0 without implying danger.
            if not isinstance(answer, dict) or answer.get("type") in (None, "noul"):
                noul.append(probabilities[name])
                noul_names.add(name)

        if not probabilities:
            return self._failure(OUTCOME_ERROR, started, raw=raw)

        engine = raw.get("engine") or raw.get("model") or self._engine
        return JudgeDecision(
            p=max(noul) if noul else max(probabilities.values()),
            answers=probabilities,
            raw=raw,
            engine=engine,
            elapsed_ms=elapsed,
            gating=tuple(k for k in probabilities if k in noul_names),
        )

    def _failure(self, error: str, started: float, *, raw: dict | None = None) -> JudgeDecision:
        elapsed = (time.monotonic() - started) * 1000.0
        logger.info(
            "JUDGE unavailable (%s) after %.0fms — allowing (fail-open)", error, elapsed,
        )
        return JudgeDecision(
            p=0.0, raw=raw, error=error, engine=self._engine, elapsed_ms=elapsed,
        )


# ── Config seam ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgeConfig:
    """Parsed ``judge:`` config block."""

    enabled: bool = False
    url: str = "http://127.0.0.1:8044"
    engine: str = "decider-0.8b"
    timeout_ms: int = 500
    consult_when: str = "deterministic-allow"
    graded_blocks: bool = False
    rules: tuple[str, ...] = ("command_safety",)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> JudgeConfig:
        if not isinstance(cfg, dict):
            return cls()
        rules = cfg.get("rules")
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            url=str(cfg.get("url", "http://127.0.0.1:8044")),
            engine=str(cfg.get("engine", "decider-0.8b")),
            timeout_ms=int(cfg.get("timeout_ms", 500)),
            consult_when=str(cfg.get("consult_when", "deterministic-allow")),
            graded_blocks=bool(cfg.get("graded_blocks", False)),
            rules=tuple(rules) if isinstance(rules, list) else ("command_safety",),
        )


def build_judge(config: dict | None) -> DecisionClient | None:
    """Build a judge client from config. Returns None when disabled.

    None (not NullDecisionClient) is the default so rules take their
    byte-identical deterministic path and cost nothing.
    """
    jc = JudgeConfig.from_dict(config)
    if not jc.enabled:
        return None
    if jc.consult_when != "deterministic-allow":
        # Only the "patterns matched nothing" seam is implemented. Accept the
        # value but say so: silently pretending to support "always"/"grey-zone"
        # would be a false claim about what runs.
        logger.warning(
            "judge.consult_when=%r is not implemented; using 'deterministic-allow'",
            jc.consult_when,
        )
    return HTTPDecisionClient(url=jc.url, engine=jc.engine, timeout_ms=jc.timeout_ms)
