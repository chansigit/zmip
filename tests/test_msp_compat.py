"""Both published MSP 0.3.0 and its public-wrapper upgrade remain usable."""

from types import SimpleNamespace

import pytest

from zmip import msp_compat


@pytest.mark.parametrize("public_available", [False, True])
def test_resolve_prefers_public_api_and_only_falls_back_when_missing(monkeypatch, public_available):
    public, legacy = object(), object()
    modules = {
        "public": SimpleNamespace(api=public) if public_available else SimpleNamespace(),
        "legacy": SimpleNamespace(_api=legacy),
    }
    requested = []

    def import_module(name):
        requested.append(name)
        return modules[name]

    monkeypatch.setattr(msp_compat.importlib, "import_module", import_module)
    result = msp_compat._resolve("public", "api", "legacy", "_api")
    assert result is (public if public_available else legacy)
    assert requested == (["public"] if public_available else ["public", "legacy"])


def test_resolve_does_not_hide_broken_public_module(monkeypatch):
    def broken_import(name):
        raise ImportError("broken dependency")

    monkeypatch.setattr(msp_compat.importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match="broken dependency"):
        msp_compat._resolve("public", "api", "legacy", "_api")
