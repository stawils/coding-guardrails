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

#: Exact commit cg builds against. Has the gemma-4 tool-call fix.
PINNED_COMMIT = "afcda09d154a285cd366135f98ffc1d357f7ddbd"
PINNED_SHORT = "afcda09d1"


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
