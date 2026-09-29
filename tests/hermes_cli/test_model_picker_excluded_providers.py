"""Tests that ``model_catalog.excluded_providers`` hides providers from the
interactive ``hermes model`` CLI picker.

The CLI picker (``hermes_cli.main.select_provider_and_model``) builds its
provider menu from ``CANONICAL_PROVIDERS`` via ``group_providers`` — a
separate code path from ``list_authenticated_providers``. These tests
verify the exclusion config is honored there too, matching the
gateway/TUI picker behavior.
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a minimal config."""
    home = tmp_path / "hermes"
    home.mkdir()
    config_yaml = home / "config.yaml"
    config_yaml.write_text("model: old-model\ncustom_providers: []\n")
    env_file = home / ".env"
    env_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


def _write_config(home, **top_level):
    import hermes_yaml as yaml
    cfg = {"model": "old-model", "custom_providers": []}
    cfg.update(top_level)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _capture_provider_labels(config_home):
    """Drive ``select_provider_and_model`` and return the provider-menu labels
    shown to the user (the first ``_prompt_provider_choice`` call). Cancels
    immediately after capturing."""
    from hermes_cli.main import select_provider_and_model

    captured: dict = {}

    def _capture_and_cancel(labels, default=0, title=None):
        # Only capture the top-level provider menu (the first call).
        if "labels" not in captured:
            captured["labels"] = list(labels)
        return None  # cancel

    with patch("hermes_cli.main._prompt_provider_choice",
               side_effect=_capture_and_cancel), \
         patch("builtins.print"):
        select_provider_and_model()

    return captured.get("labels", [])


def test_cli_picker_hides_excluded_provider(config_home):
    """``excluded_providers: [openrouter]`` must remove the OpenRouter row
    from the ``hermes model`` provider menu."""
    _write_config(config_home, **{"model_catalog": {"excluded_providers": ["openrouter"]}})

    labels = _capture_provider_labels(config_home)
    assert labels, "provider menu was empty"
    assert not any("OpenRouter" in lbl for lbl in labels), (
        f"OpenRouter should be hidden by excluded_providers, got: {labels}"
    )


def test_cli_picker_hides_excluded_provider_by_alias(config_home):
    """Exclusion by an alias (not the canonical slug) must also hide the
    provider, matching ``list_authenticated_providers``' matching against
    hermes_id / alias names."""
    # 'openai' is an alias-style hermes id; ensure excluding it hides the
    # canonical openai provider row if present. Use the canonical slug's
    # alias from _PROVIDER_ALIASES to stay robust to renames.
    from hermes_cli.models import _PROVIDER_ALIASES, CANONICAL_PROVIDERS

    # Find a canonical provider that has at least one alias and is a leaf
    # row (not folded into a multi-member group) so its label appears
    # directly. Pick the first such provider.
    target_slug = None
    target_alias = None
    for alias, canon in _PROVIDER_ALIASES.items():
        if canon and any(p.slug == canon for p in CANONICAL_PROVIDERS):
            target_slug = canon
            target_alias = alias
            break
    if target_slug is None:
        pytest.skip("no aliased canonical provider available to test")

    from hermes_cli.models import _PROVIDER_LABELS
    target_label_fragment = _PROVIDER_LABELS.get(target_slug, target_slug)

    # Baseline: the provider appears without exclusion.
    _write_config(config_home)
    baseline = _capture_provider_labels(config_home)
    assert any(target_label_fragment in lbl for lbl in baseline), (
        f"sanity: {target_slug} ({target_label_fragment!r}) should appear by "
        f"default; labels={baseline}"
    )

    # Excluding by alias hides it.
    _write_config(
        config_home,
        **{"model_catalog": {"excluded_providers": [target_alias]}},
    )
    excluded_labels = _capture_provider_labels(config_home)
    assert not any(target_label_fragment in lbl for lbl in excluded_labels), (
        f"excluding alias {target_alias!r} should hide {target_slug}; "
        f"labels={excluded_labels}"
    )


@pytest.mark.parametrize("explicit_only,include_unconfigured", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("current_provider", ["anthropic", "moa"])
def test_inventory_exclusions_survive_setup_and_virtual_rows(
    config_home, tmp_path, monkeypatch, explicit_only, include_unconfigured, current_provider,
):
    from hermes_cli.inventory import build_models_payload, load_picker_context
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    other_home = tmp_path / "other"
    other_home.mkdir()
    config = {
        "model": {"provider": current_provider, "default": "saved-model"},
        # MoA is opt-in: an enabled preset keeps its virtual row visible unless excluded.
        "moa": {"presets": {"default": {"enabled": True}}},
    }
    _write_config(config_home, **config, model_catalog={"excluded_providers": [" CLAUDE ", " MoA "]})
    _write_config(other_home, **config)
    # No credentials: the current row comes from recovery and all others from setup.
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **kw: [])
    for home, excluded in [(config_home, True), (other_home, False), (config_home, True)]:
        token = set_hermes_home_override(home)
        try:
            payload = build_models_payload(
                load_picker_context(), explicit_only=explicit_only,
                include_unconfigured=include_unconfigured,
            )
        finally:
            reset_hermes_home_override(token)
        slugs = {row["slug"] for row in payload["providers"]}
        if excluded:
            assert not slugs & {"anthropic", "moa"}
        else:
            assert current_provider in slugs
            if include_unconfigured:
                assert "anthropic" in slugs
                if not explicit_only:
                    assert "moa" in slugs
        assert payload["provider"] == current_provider
        assert payload["model"] == "saved-model"


@pytest.mark.parametrize("current_provider", ["anthropic", "moa"])
def test_gateway_picker_cannot_reintroduce_excluded_moa(config_home, monkeypatch, current_provider):
    from hermes_cli.model_switch_providers import list_picker_providers

    _write_config(config_home, moa={"presets": {"default": {"enabled": True}}})
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **kw: [])
    for excluded in ([], [" MoA "], []):
        rows = list_picker_providers(
            current_provider=current_provider, include_moa=True, excluded_providers=excluded,
        )
        assert ("moa" in {row["slug"] for row in rows}) is (not bool(excluded))
