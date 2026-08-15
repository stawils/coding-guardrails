"""Vision captioning tests — image_url blocks become [image: caption] text.

Covers the pure block-replacement logic in handler.caption_images_in_messages
with an injectable captioner (no HTTP). Failure degrades to a placeholder so
requests never die on a text-only backend.
"""

from __future__ import annotations

import pytest

from coding_guardrails.proxy.handler import (
    _image_urls_in,
    caption_images_in_messages,
)


async def _fake_captioner(url: str) -> str:
    return f"caption of {url[:16]}"


async def _empty_captioner(url: str) -> str:
    return ""


@pytest.mark.asyncio
async def test_image_urls_in_extracts_urls():
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "http://x/img.png"}},
    ]
    urls = _image_urls_in(content)
    assert urls == ["data:image/png;base64,AAAA", "http://x/img.png"]


def test_image_urls_in_non_list():
    assert _image_urls_in("plain text") == []
    assert _image_urls_in(None) == []
    assert _image_urls_in([{"type": "text", "text": "no images"}]) == []


@pytest.mark.asyncio
async def test_image_replaced_with_caption():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }]
    out = await caption_images_in_messages(msgs, object(), captioner=_fake_captioner)
    assert out[0]["role"] == "user"
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "What is in this image?"}
    assert blocks[1]["type"] == "text"
    assert blocks[1]["text"] == "[image: caption of data:image/png;b]"


@pytest.mark.asyncio
async def test_caption_failure_degrades_to_placeholder():
    msgs = [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "http://x/a.png"}}],
    }]
    out = await caption_images_in_messages(msgs, object(), captioner=_empty_captioner)
    assert out[0]["content"][0]["text"] == "[image: (unavailable)]"


@pytest.mark.asyncio
async def test_multi_image_message_captions_each():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:a"}},
            {"type": "image_url", "image_url": {"url": "data:b"}},
        ],
    }]
    out = await caption_images_in_messages(msgs, object(), captioner=_fake_captioner)
    texts = [b["text"] for b in out[0]["content"]]
    assert texts == [
        "[image: caption of data:a]",
        "[image: caption of data:b]",
    ]


@pytest.mark.asyncio
async def test_messages_without_images_untouched():
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "plain text"},
        {"role": "user", "content": [{"type": "text", "text": "multi-block"}]},
    ]
    out = await caption_images_in_messages(msgs, object(), captioner=_fake_captioner)
    assert out == msgs


@pytest.mark.asyncio
async def test_empty_url_becomes_placeholder():
    msgs = [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": ""}}],
    }]
    out = await caption_images_in_messages(msgs, object(), captioner=_fake_captioner)
    assert out[0]["content"][0]["text"] == "[image: (unavailable)]"
