"""Unit tests for src/coding_guardrails/server/sources.py."""

import pytest

from coding_guardrails.server.sources import (
    DRAFTERS,
    SOURCES,
    DrafterSource,
    ModelSource,
    get_drafter,
    get_source,
)


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


class TestDrafterSources:
    """Drafters are speculative-decoding GGUFs paired with a target profile."""

    def test_defaults_and_lookup(self) -> None:
        spec = DrafterSource(
            repo_id="a/b", filename="d.gguf", spec_type="draft-dflash", n_max=4,
            vram_gb=1.0,
        )
        assert spec.license == ""
        assert get_drafter("nope") is None

    def test_every_drafter_is_well_formed(self) -> None:
        for name, spec in DRAFTERS.items():
            assert spec.repo_id, f"repo_id empty for {name}"
            assert spec.filename.endswith(".gguf"), f"bad filename for {name}"
            assert spec.spec_type.startswith("draft-"), f"bad spec_type for {name}"
            assert 1 <= spec.n_max <= 32, f"n_max out of range for {name}"
            assert 0 < spec.vram_gb < 8, f"implausible vram_gb for {name}"
            assert get_drafter(name) is spec

    def test_profile_drafter_references_a_known_entry(self) -> None:
        from coding_guardrails.models.profiles import PROFILES

        for profile in PROFILES.values():
            if profile.drafter is None:
                continue
            assert profile.drafter in DRAFTERS, (
                f"{profile.name} references unregistered drafter {profile.drafter!r}"
            )
