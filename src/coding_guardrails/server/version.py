"""Pinned llama.cpp version.

The pin is chosen to include the Gemma 4 specialized tool-call parser
(llama.cpp #21418, #21704). Older builds (including LM Studio's bundled
binary) corrupt complex/nested JSON tool-call arguments -- see issue #21680.

Updating the pin: bump PINNED_COMMIT to a newer commit, confirm the desired
fix commits are ancestors, then ``cg server build`` re-checks out and rebuilds.
"""

from __future__ import annotations

import logging
import subprocess

from coding_guardrails.server.paths import binary_path, source_dir

logger = logging.getLogger("coding_guardrails.server.version")

#: Upstream repo the pin is fetched from. Currently a **fork**: DSpine
#: speculative decoding (Tencent Hunyuan, arXiv 2609.36173) is not upstream
#: yet. The fork's `dspine` branch is ggml-org master + exactly one commit
#: (`dflash : add DSpine draft model support`, 474 lines: gguf tensor names,
#: the code-bridge graph, and the serial last-write refinement in
#: src/models/dflash.cpp — no network/shell/file-write additions).
#: Switch back to https://github.com/ggml-org/llama.cpp.git once upstream
#: lands DSpine; the only thing cg depends on is that commit being an ancestor.
PINNED_REPO = "https://github.com/Ajay9o9/llama.cpp.git"

#: Exact commit cg builds against. ggml-org master 0.9.0-dev (2026-11-08) +
#: the DSpine commit. Keeps everything the previous pin carried (gemma-4
#: tool-call fix, qwen35/flint/lfm3 archs) and adds DSpine draft support.
#: Build 2026-11-09.
PINNED_COMMIT = "d493f841b20241606c47a4e0537d3c786b5cf50b"
PINNED_SHORT = "d493f84"


def installed_commit() -> str | None:
    """Full SHA the checked-out source is at, or None if not a checkout."""
    src = source_dir()
    if not (src / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return None


def binary_version() -> str | None:
    """Version string the compiled binary reports, or None if missing/broken.

    llama-server prints a line like ``version: 9284 (afcda09d1)``.
    """
    binary = binary_path()
    if not binary.exists():
        return None
    try:
        out = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    text = out.stdout + out.stderr
    for line in text.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return text.strip()[:120] or None


def is_up_to_date() -> bool:
    """True if the source checkout matches the pinned commit."""
    commit = installed_commit()
    return commit is not None and commit.startswith(PINNED_COMMIT)
