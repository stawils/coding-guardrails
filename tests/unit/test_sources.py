"""Unit tests for src/coding_guardrails/server/sources.py."""

import pytest

from coding_guardrails.server.sources import (
    DRAFTERS,
    LORAS,
    SOURCES,
    DrafterSource,
    LoraSource,
    ModelSource,
    get_drafter,
    get_lora,
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


class TestLoraSources:
    """LoRA adapters are GGUFs applied over a specific target profile."""

    def test_defaults_and_lookup(self) -> None:
        spec = LoraSource(
            repo_id="a/b", filename="l.gguf", base_profile="p", vram_gb=0.3,
        )
        assert spec.license == ""
        assert spec.notes == ""
        assert get_lora("nope") is None

    def test_every_lora_is_well_formed(self) -> None:
        for name, spec in LORAS.items():
            assert spec.repo_id, f"repo_id empty for {name}"
            assert spec.filename.endswith(".gguf"), f"bad filename for {name}"
            assert spec.base_profile, f"base_profile empty for {name}"
            assert 0 < spec.vram_gb < 4, f"implausible vram_gb for {name}"
            assert get_lora(name) is spec

    def test_profile_lora_references_a_known_entry(self) -> None:
        from coding_guardrails.models.profiles import PROFILES

        for profile in PROFILES.values():
            if getattr(profile, "lora", None) is None:
                continue
            assert profile.lora in LORAS, (
                f"{profile.name} references unregistered lora {profile.lora!r}"
            )

    def test_lora_base_profile_exists(self) -> None:
        """An adapter is useless on the wrong base — the pairing must be real."""
        from coding_guardrails.models.profiles import PROFILES

        for name, spec in LORAS.items():
            assert spec.base_profile in PROFILES, (
                f"{name} declares base_profile {spec.base_profile!r}, which is "
                "not a registered profile"
            )

    def test_lora_serving_profile_shares_its_base_gguf(self) -> None:
        """The adapted profile must resolve to the adapter's base GGUF."""
        from coding_guardrails.models.profiles import PROFILES

        for profile in PROFILES.values():
            lora_id = getattr(profile, "lora", None)
            if lora_id is None:
                continue
            base = LORAS[lora_id].base_profile
            adapted_src = get_source(profile.name)
            base_src = get_source(base)
            assert adapted_src is not None and base_src is not None
            assert (adapted_src.repo_id, adapted_src.filename) == (
                base_src.repo_id,
                base_src.filename,
            ), f"{profile.name} must download the same GGUF as {base}"
