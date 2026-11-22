"""Unit tests for the usage-carrying boundary in the proxy.

Regression: the proxy re-serializes the response instead of relaying the
backend body, so `usage` was dropped and every client saw 0/0/0 — while the
subagent token accounting read exactly those zeros.
"""
from types import SimpleNamespace

from coding_guardrails.proxy.handler import attach_usage


def _payload(usage=None):
    body = {"id": "chatcmpl-x", "object": "chat.completion", "choices": [{"index": 0}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _client(usage):
    return SimpleNamespace(last_usage=usage)


class TestAttachUsage:
    def test_zero_usage_replaced_from_client(self):
        client = _client({0: SimpleNamespace(prompt_tokens=17, completion_tokens=5, total_tokens=22)})
        out = attach_usage(_payload({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}), client)
        assert out["usage"] == {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22}

    def test_missing_usage_key_added(self):
        client = _client({0: SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)})
        out = attach_usage(_payload(), client)
        assert out["usage"]["total_tokens"] == 7

    def test_dict_usage_objects_accepted(self):
        client = _client({0: {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})
        out = attach_usage(_payload(), client)
        assert out["usage"]["total_tokens"] == 3

    def test_existing_nonzero_usage_is_not_overwritten(self):
        client = _client({0: SimpleNamespace(prompt_tokens=99, completion_tokens=99, total_tokens=198)})
        keep = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        out = attach_usage(_payload(keep), client)
        assert out["usage"] == keep

    def test_no_capture_leaves_zeros(self):
        out = attach_usage(_payload({"total_tokens": 0}), _client({}))
        assert out["usage"] == {"total_tokens": 0}

    def test_client_without_attribute_is_safe(self):
        """No capture seam at all: payload passes through with no usage key."""
        out = attach_usage(_payload(), SimpleNamespace())
        assert "usage" not in out

    def test_total_derived_when_missing(self):
        client = _client({0: SimpleNamespace(prompt_tokens=8, completion_tokens=2)})
        out = attach_usage(_payload(), client)
        assert out["usage"]["total_tokens"] == 10

    def test_cache_counters_surfaced(self):
        client = _client({0: SimpleNamespace(
            prompt_tokens=5, completion_tokens=1, total_tokens=6, cache_read_input_tokens=40)})
        out = attach_usage(_payload(), client)
        assert out["usage"]["prompt_tokens_details"] == {"cached_tokens": 40}

    def test_non_dict_payload_untouched(self):
        events = [{"data": "x"}]
        assert attach_usage(events, _client({0: SimpleNamespace(prompt_tokens=1)})) is events

    def test_all_zero_capture_does_not_invent_counts(self):
        client = _client({0: SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)})
        out = attach_usage(_payload({"total_tokens": 0}), client)
        assert out["usage"] == {"total_tokens": 0}
