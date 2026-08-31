"""Per-profile context-budget wiring tests.

Measured 2026-08-31: Qwen3.8-27B tool-calling collapses to prose at
~20-27K prompt tokens through the proxy; solid below ~11K, flaky at
13-19K regardless of temperature. Profile.context_budget (None = use the
--context-budget global default) lets per-model measurements slot in.
"""

from __future__ import annotations

from coding_guardrails.models.profiles import get_profile


def test_context_budget_field_defaults_to_none() -> None:
    # A profile without a measured budget must fall back to the global default.
    p = get_profile("LFM2.5-2.6B-BF16")
    assert p is not None
    assert p.context_budget is None


def test_qwen38_has_measured_budget() -> None:
    p = get_profile("Qwen3.8-27B-UD-Q3_K_XL")
    assert p is not None
    assert p.context_budget == 12000
    assert p.context_budget < p.context_tokens


def test_budget_is_well_below_context_window() -> None:
    """A set context_budget must always be < the model's context window."""
    profs = [
        get_profile("Qwen3.8-27B-UD-Q3_K_XL"),
        get_profile("Qwen3.6-27B-UD-Q4_K_XL"),
        get_profile("LFM2.5-2.6B-BF16"),
        get_profile("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL"),
    ]
    for p in profs:
        assert p is not None
        if p.context_budget is not None:
            assert p.context_budget < p.context_tokens