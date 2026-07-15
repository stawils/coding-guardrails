"""Unit tests for src/coding_guardrails/server/sources.py."""

import pytest

from coding_guardrails.server.sources import ModelSource, get_source, SOURCES


class TestModelSourceDefaults:
    """Test that ModelSource defaults work correctly."""

    def test_license_defaults_to_empty_string(self) -> None:
        """ModelSource.license defaults to "" when omitted."""
        source = ModelSource(repo_id="test/repo", filename="model.gguf")
        assert source.license == ""


class TestGetSourceKnownProfile:
    """Test that get_source() returns the correct ModelSource for known profiles."""

    @pytest.mark.parametrize("profile_name", list(SOURCES.keys()))
    def test_returns_source_for_known_profile(self, profile_name: str) -> None:
        """get_source() returns a ModelSource for each known profile name."""
        source = get_source(profile_name)
        assert source is not None, f"Expected source for {profile_name}"
        assert source.repo_id == SOURCES[profile_name].repo_id
        assert source.filename == SOURCES[profile_name].filename


class TestGetSourceUnknownProfile:
    """Test that get_source() returns None for unknown profile names."""

    def test_returns_none_for_unknown_profile(self) -> None:
        """get_source() returns None for profile names not in SOURCES."""
        assert get_source("nonexistent-profile") is None
        assert get_source("") is None
        assert get_source("Qwen3.5-9B-UD-Q4_K_XL-variant") is None


class TestSOURCESIntegrity:
    """Test that every entry in SOURCES has required non-empty fields."""

    def test_all_entries_have_non_empty_repo_id(self) -> None:
        """Every ModelSource in SOURCES has a non-empty repo_id."""
        for name, source in SOURCES.items():
            assert source.repo_id, f"repo_id is empty for {name}"
            assert len(source.repo_id) > 0

    def test_all_entries_have_non_empty_filename(self) -> None:
        """Every ModelSource in SOURCES has a non-empty filename."""
        for name, source in SOURCES.items():
            assert source.filename, f"filename is empty for {name}"
            assert len(source.filename) > 0

    def test_source_count(self) -> None:
        """Verify the number of registered sources."""
        assert len(SOURCES) > 0
