"""Extract file writes hidden inside shell commands.

An agent that writes files with `cat > f <<'EOF' ... EOF` never touches the
`write`/`edit` tools, so every edit-class rule silently skips it: prerequisites
(read-before-edit) sees no edit, lint never runs, dup_write never hashes the
content, and path/sensitive-file checks only ever saw the command string.
Measured 2026-11-22: an Inkling 2 Mini worker wrote both of its files this way
and 0 of 3 edit-class rules applied (plans/2026-11-22_inkling-worker-eval.md).

This module turns a command string into the writes it performs, so the
middleware can re-check them as edit-class actions. Deliberately conservative:
it recognizes common shapes and returns nothing when unsure. A false negative
costs coverage on an exotic spelling; a false positive would block a legitimate
command, which is worse.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

#: Redirect targets that are never real file writes.
_IGNORED_TARGETS = {
    "/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/zero",
    "/dev/random", "/dev/urandom",
}

#: `cat > file`, `cat >> file`, `cat <<'EOF' > file`, ...
_HEREDOC_RE = re.compile(
    r"<<-?\s*(?P<q>['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)(?P=q)"
)

#: A redirect appearing anywhere in a segment: `>`, `>>`, `&>`, `&>>` + target.
_REDIRECT_RE = re.compile(
    r"(?<![0-9<>&])(?P<op>&>>?|>>?)\s*(?P<target>\"[^\"]+\"|'[^']+'|[^\s;&|<>]+)"
)

#: Tools whose output is a file write when combined with a redirect.
_WRITER_RE = re.compile(r"\b(?:cat|tee|printf|echo)\b")

#: `tee [-a] path` (optionally with a heredoc feeding it).
_TEE_RE = re.compile(
    r"\btee\b(?P<flags>(?:\s+-[a-zA-Z]+)*)\s+(?P<target>\"[^\"]+\"|'[^']+'|[^\s;&|<]+)"
)


@dataclass(frozen=True)
class ShellWrite:
    """A file write performed by a shell command."""

    path: str
    content: str | None  # None = content not visible in the command (e.g. `> f`)
    append: bool = False


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _is_ignored(path: str) -> bool:
    p = path.strip()
    if not p:
        return True
    if p in _IGNORED_TARGETS:
        return True
    # Process substitution / fd duplication leftovers.
    if p.startswith("(") or p.startswith("&") or p.isdigit():
        return True
    return False


def _heredoc_bodies(command: str) -> list[tuple[str, str]]:
    """Return [(tag, body)] for each heredoc in the command.

    Bodies are taken verbatim between the marker line and a line that is exactly
    the terminator (leading tabs stripped for `<<-`).
    """
    out: list[tuple[str, str]] = []
    for match in _HEREDOC_RE.finditer(command):
        tag = match.group("tag")
        rest = command[match.end():]
        if rest.startswith("\n"):
            rest = rest[1:]
        else:
            # Marker not at end of line: `cat <<EOF > f` — body starts after
            # the newline that follows the segment.
            nl = rest.find("\n")
            if nl == -1:
                continue
            rest = rest[nl + 1:]
        body_lines: list[str] = []
        for line in rest.splitlines():
            if line.strip() == tag:
                out.append((tag, "\n".join(body_lines)))
                break
            body_lines.append(line.lstrip("\t") if match.group(0).startswith("<<-") else line)
    return out


def _iter_segments(command: str) -> list[str]:
    """Split a command into simple segments on ; && || | and newlines.

    Heredoc bodies are intentionally left in place: segment splitting runs on the
    full string, and redirect detection is anchored to the line that carries the
    `cat`/`tee`/`printf` command, so the body never contributes a spurious
    redirect for these shapes.
    """
    return [seg for seg in re.split(r"&&|\|\||;|\n", command) if seg.strip()]


def extract_writes(command: str) -> list[ShellWrite]:
    """Return the file writes a shell command performs.

    Recognizes (per segment):
      - `cat > f <<'EOF' ...` / `cat >> f <<EOF ...` / `cat <<EOF > f`
      - `tee f <<EOF`, `tee -a f <<EOF`
      - `echo ... > f`, `printf ... > f`, `printf ... >> f`
      - bare `> f` truncation and `>> f` append
    Returns [] when nothing recognizable is found.
    """
    if not command or (">" not in command and "<<" not in command):
        return []

    bodies = _heredoc_bodies(command)
    writes: list[ShellWrite] = []

    for segment in _iter_segments(command):
        redirects = [
            (m.group("op"), _strip_quotes(m.group("target")))
            for m in _REDIRECT_RE.finditer(segment)
        ]
        redirects = [(op, tgt) for op, tgt in redirects if not _is_ignored(tgt)]

        # `tee [-a] file <<EOF` writes without any `>` redirect.
        tee = _TEE_RE.search(segment)
        if tee is not None and "<<" in segment:
            target = _strip_quotes(tee.group("target"))
            if not _is_ignored(target):
                flags = tee.group("flags") or ""
                append = bool(re.search(r"-\w*a", flags))
                writes.append(ShellWrite(
                    path=target,
                    content=bodies[0][1] if bodies else None,
                    append=append,
                ))

        if not redirects:
            continue

        has_writer = bool(_WRITER_RE.search(segment))
        for op, target in redirects:
            append = op.endswith(">>")
            content: str | None = None
            if has_writer and bodies:
                # First heredoc body belongs to the first heredoc on this line.
                content = bodies[0][1]
            elif not has_writer:
                # Bare `> f` truncation — a write with invisible content.
                content = ""
            writes.append(ShellWrite(path=target, content=content, append=append))

    # Deduplicate by (path, append) keeping the first — `cat > f <<EOF` yields
    # one redirect and must not be reported twice.
    seen: set[tuple[str, bool]] = set()
    unique: list[ShellWrite] = []
    for w in writes:
        key = (w.path, w.append)
        if key in seen:
            continue
        seen.add(key)
        unique.append(w)
    return unique


def shell_tool_args(args: dict, command_args: tuple[str, ...] = ("command", "cmd", "script")) -> str:
    """Return the command string from a tool call's args, or ''."""
    for name in command_args:
        value = args.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _shlex_quote_for_log(path: str) -> str:  # pragma: no cover - helper for tests
    return shlex.quote(path)
