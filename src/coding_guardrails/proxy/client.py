"""Extended Forge client that passes max_tokens through to the backend.

Forge's LlamafileClient._apply_sampling() only handles its own field list
(temperature, top_p, etc.) — it deliberately ignores max_tokens. This
subclass extends it to also forward max_tokens / n_predict, which prevents
runaway model generation.

This is the only place we extend Forge behavior without modifying the
installed package. Everyone gets the fix just by pip-installing our package
and Forge — no manual edits required.
"""

from __future__ import annotations

from typing import Any

from forge.clients.llamafile import LlamafileClient


class SafeLlamafileClient(LlamafileClient):
    """LlamafileClient that forwards max_tokens to the backend.

    Adds max_tokens, max_completion_tokens, and n_predict to the sampling
    pipeline so the backend always receives an output cap.

    The default_max_tokens constructor arg is injected when the caller
    doesn't provide one — acts as a safety net against runaway generation.
    """

    _EXTRA_SAMPLING_FIELDS = ("max_tokens", "n_predict")

    def __init__(self, *args: Any, default_max_tokens: int = 8192, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._default_max_tokens = default_max_tokens

    def _apply_sampling(
        self, body: dict[str, Any], sampling: dict[str, Any] | None = None,
    ) -> None:
        # Let Forge handle its own fields first
        super()._apply_sampling(body, sampling)

        # Forward our extra fields (max_tokens, n_predict)
        for field in self._EXTRA_SAMPLING_FIELDS:
            override = (sampling or {}).get(field)
            if override is not None:
                body[field] = override
                break  # First match wins (max_tokens preferred over n_predict)

        # Safety net: if nobody set any cap, inject a default
        if "max_tokens" not in body and "n_predict" not in body:
            body["max_tokens"] = self._default_max_tokens
