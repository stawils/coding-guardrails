"""Shell-command write extraction + edit-class coverage.

An agent writing files via `cat > f <<'EOF'` bypasses every write/edit rule.
Measured 2026-11-22 on an Inkling 2 Mini worker: both files written by heredoc,
0 of 3 edit-class rules applied. These tests pin the coverage in both
directions: dangerous shell writes must be judged, benign ones must not be.
"""
import pytest

from coding_guardrails.middleware import CodingGuardrails
from coding_guardrails.rules.base import Action, ToolCall
from coding_guardrails.rules.bash_writes import extract_writes


class TestExtraction:
    def test_heredoc_write_is_extracted_with_body(self):
        cmd = "mkdir -p /tmp/x && cat > /tmp/x/a.py <<'EOF'\nx = 1\nEOF"
        writes = extract_writes(cmd)
        assert len(writes) == 1
        assert writes[0].path == "/tmp/x/a.py"
        assert writes[0].content == "x = 1"
        assert writes[0].append is False

    def test_append_variant(self):
        writes = extract_writes("cat >> log.txt <<'EOT'\nmore\nEOT")
        assert writes[0].append is True
        assert writes[0].path == "log.txt"

    def test_unquoted_and_redirect_after_marker(self):
        writes = extract_writes("cat <<EOF > target.txt\nbody\nEOF")
        assert writes[0].path == "target.txt"
        assert writes[0].content == "body"

    def test_tee_heredoc(self):
        writes = extract_writes("tee -a notes.md <<'EOF'\nbody\nEOF")
        assert writes[0].path == "notes.md"
        assert writes[0].append is True

    @pytest.mark.parametrize("cmd", [
        "ls -la /tmp 2>&1",          # fd duplication, not a write
        "rm -rf build/",             # no write
        "cat /tmp/x.py",             # read
        "cmd > /dev/null",           # discarded
        "python3 - <<'PY'\nprint(1)\nPY",   # stdin script, no file target
    ])
    def test_non_writes_return_nothing(self, cmd):
        assert extract_writes(cmd) == []


class TestEditClassCoverage:
    """The middleware must judge shell writes through the edit rules."""

    def _gw(self):
        return CodingGuardrails.defaults()

    def test_heredoc_into_git_config_blocks(self):
        gw = self._gw()
        call = ToolCall(tool="bash", args={
            "command": "cat > .git/config <<'EOF'\n[core]\nEOF"})
        res = gw.check([call])
        assert res.has_blocks, "heredoc write into .git/ must be blocked"
        names = {b.rule_name for b in res.blocked}
        assert "sensitive_files" in names

    def test_heredoc_into_ci_workflow_blocks(self):
        gw = self._gw()
        call = ToolCall(tool="bash", args={
            "command": "cat > .github/workflows/ci.yml <<'EOF'\nname: x\nEOF"})
        res = gw.check([call])
        assert res.has_blocks
        assert {b.rule_name for b in res.blocked} == {"sensitive_files"}

    def test_heredoc_into_system_file_blocks(self):
        gw = self._gw()
        call = ToolCall(tool="bash", args={"command": "cat > /etc/passwd <<'EOF'\nx\nEOF"})
        res = gw.check([call])
        assert res.has_blocks
        assert "path_safety" in {b.rule_name for b in res.blocked}

    def test_plain_bash_without_write_is_untouched(self):
        gw = self._gw()
        res = gw.check([ToolCall(tool="bash", args={"command": "ls -la /tmp 2>&1"})])
        assert not res.has_blocks

    def test_benign_heredoc_write_is_allowed(self):
        gw = self._gw()
        call = ToolCall(tool="bash", args={
            "command": "mkdir -p /tmp/scratch && cat > /tmp/scratch/new_file_zzz.py <<'EOF'\nx = 1\nEOF"})
        res = gw.check([call])
        assert not res.has_blocks, [b.rule_name for b in res.blocked]

    def test_command_rules_do_not_double_fire(self):
        gw = self._gw()
        call = ToolCall(tool="bash", args={
            "command": "cat > /etc/passwd <<'EOF'\nx\nEOF"})
        res = gw.check([call])
        rule_names = [b.rule_name for b in res.blocked]
        assert rule_names.count("command_safety") <= 1

    def test_record_advances_dup_write_from_heredoc(self):
        """Two identical heredoc writes must register as a duplicate write."""
        gw = self._gw()
        call = ToolCall(tool="bash", args={
            "command": "cat > /tmp/dup_probe_zzz.py <<'EOF'\nsame = 1\nEOF"})
        gw.record([call])          # first write recorded
        res = gw.check([call])     # second identical write checked
        names = {n.rule_name for n in res.nudges}
        assert "dup_write" in names, [n.rule_name for n in res.nudges]
