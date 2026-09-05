"""Fault-injection tests for complete output sets and runtime-aware resume."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from zmip import cache, publication, runtime


def publish_example(root, text):
    with publication.staging(root) as stage:
        for name in (*publication.DATA_FILES, "report.html"):
            (stage / name).write_text(text + name)
        publication.publish(root, stage)


def snapshot(root):
    return {name: (root / name).read_bytes() for name in (*publication.DATA_FILES, "report.html", publication.RECEIPT)}


def test_publication_exception_restores_entire_previous_set(tmp_path, monkeypatch):
    publish_example(tmp_path, "old")
    before = snapshot(tmp_path)
    original = publication.os.replace

    def interrupted(source, target):
        if Path(target) == tmp_path / "zmip_removed.csv" and Path(source).parent.name == "new":
            raise OSError("disk write failed")
        return original(source, target)

    monkeypatch.setattr(publication.os, "replace", interrupted)
    with pytest.raises(OSError, match="disk write"):
        publish_example(tmp_path, "new")
    assert snapshot(tmp_path) == before
    assert publication.complete(tmp_path)
    assert not (tmp_path / publication.JOURNAL).exists()


@pytest.mark.parametrize("previous", [True, False])
def test_hard_exit_is_recovered_before_results_are_trusted(tmp_path, previous):
    if previous:
        publish_example(tmp_path, "old")
        before = snapshot(tmp_path)
    # os._exit deliberately skips all finally blocks, as SIGKILL would.
    script = """
import os, sys
from pathlib import Path
from zmip import publication
root = Path(sys.argv[1])
original = os.replace
def interrupted(source, target):
    if Path(target) == root / "zmip_removed.csv" and Path(source).parent.name == "new":
        os._exit(17)
    return original(source, target)
os.replace = interrupted
with publication.staging(root) as stage:
    for name in (*publication.DATA_FILES, "report.html"):
        (stage / name).write_text("new " + name)
    publication.publish(root, stage)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False)
    assert result.returncode == 17
    assert not publication.complete(tmp_path)
    with pytest.raises(RuntimeError, match="incomplete"):
        publication.require_complete(tmp_path)
    publication.recover(tmp_path)
    publication.recover(tmp_path)
    if previous:
        assert snapshot(tmp_path) == before and publication.complete(tmp_path)
    else:
        assert all(not (tmp_path / name).exists() for name in publication.DATA_FILES)
        assert not (tmp_path / publication.RECEIPT).exists()


def test_report_can_be_rebuilt_but_damaged_data_cannot(tmp_path):
    publish_example(tmp_path, "old")
    (tmp_path / "report.html").write_text("damaged report")
    assert not publication.complete(tmp_path)
    publication.require_complete(tmp_path)
    publication.refresh_report_receipt(tmp_path, tmp_path / "report.html")
    assert publication.complete(tmp_path)
    (tmp_path / "zmip_removed.csv").write_text("damaged data")
    with pytest.raises(RuntimeError, match="incomplete"):
        publication.require_complete(tmp_path)


def test_real_report_rebuild_refreshes_receipt_without_changing_data(tmp_path):
    from zmip.report import generate_report

    publish_example(tmp_path, "old")
    before = {name: (tmp_path / name).read_bytes() for name in publication.DATA_FILES}
    generate_report(tmp_path)
    assert "<!DOCTYPE html>" in (tmp_path / "report.html").read_text()
    assert publication.complete(tmp_path)
    assert all((tmp_path / name).read_bytes() == data for name, data in before.items())


def test_runtime_change_invalidates_resume_even_with_identical_input(tmp_path, monkeypatch):
    source = tmp_path / "input.h5ad"
    source.write_bytes(b"input")
    monkeypatch.setattr(cache, "runtime_identity", lambda: {"versions": {"msp-sc": "0.2.0"}, "source": "first"})
    generation = cache.prepare_run(tmp_path / "out", source, {})
    monkeypatch.setattr(cache, "runtime_identity", lambda: {"versions": {"msp-sc": "0.2.0"}, "source": "changed"})
    with pytest.raises(ValueError, match="runtime/code changed"):
        cache.prepare_run(tmp_path / "out", source, {})
    assert cache.prepare_run(tmp_path / "out", source, {}, force=True) != generation


def test_source_identity_is_path_independent_and_not_only_a_version(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    for root in (left, right):
        root.mkdir()
        (root / "module.py").write_text("x = 1\n")
    assert runtime.source_digest(left) == runtime.source_digest(right)
    (right / "module.py").write_text("x = 2\n")
    assert runtime.source_digest(left) != runtime.source_digest(right)


def test_current_runtime_passes_behavioral_compatibility_check():
    runtime.check_runtime()
    identity = runtime.runtime_identity()
    assert all(name in identity["source_sha256"] for name in runtime.SOURCE_MODULES)
    json.dumps(identity)
