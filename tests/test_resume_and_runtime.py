"""Exercise resume identity, interrupted work, tool parsing and child ownership."""

import asyncio
import copy
import importlib
import json
import logging
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from zmip import cache
from zmip.cli import parse_harmony

lineage = importlib.import_module("zmip.lineage")
plan_module = importlib.import_module("zmip.plan")
annotate = importlib.import_module("zmip.annotate")


def test_resume_checks_input_contents_options_and_legacy_outputs(tmp_path):
    source = tmp_path / "input.h5ad"
    source.write_bytes(b"first input")
    root = tmp_path / "out"
    generation = cache.prepare_run(root, source, {"resolutions": (1., 2.)})
    assert cache.prepare_run(root, source, {"resolutions": [1., 2.]}) == generation
    # Agent settings are audit information: refreshed, never compared.
    assert cache.prepare_run(root, source, {"resolutions": [1., 2.]}, agent={"max_turns": 80}) == generation
    assert json.loads((root / ".zmip-run.json").read_text())["agent"] == {"max_turns": 80}
    source.write_bytes(b"other input")
    with pytest.raises(ValueError, match="input or options changed"):
        cache.prepare_run(root, source, {"resolutions": [1., 2.]})
    source.write_bytes(b"first input")
    with pytest.raises(ValueError, match="input or options changed"):
        cache.prepare_run(root, source, {"resolutions": [0.3, 1., 2.]})
    (root / ".zmip-run.json").unlink()
    (root / "zmip_plan.json").write_text("{}")
    with pytest.raises(ValueError, match="legacy"):
        cache.prepare_run(root, source, {})
    assert cache.prepare_run(root, source, {}, force=True) != generation


def test_receipts_reject_tampering_missing_files_and_new_generation(tmp_path):
    (tmp_path / "result.csv").write_text("cell\n001\n")
    files = ["result.csv"]
    cache.seal(tmp_path, "markers", "run1", files)
    assert cache.valid(tmp_path, "markers", "run1", files)
    assert not cache.valid(tmp_path, "markers", "run2", files)
    (tmp_path / "result.csv").write_text("cell\n002\n")
    assert not cache.valid(tmp_path, "markers", "run1", files)

    (tmp_path / "result.csv").unlink()
    assert not cache.valid(tmp_path, "markers", "run1", files)


def test_output_lock_rejects_a_second_writer_and_releases_on_failure(tmp_path):
    with pytest.raises(ValueError, match="interrupted"):
        with cache.lock_run(tmp_path):
            with pytest.raises(RuntimeError, match="another zmip run"):
                with cache.lock_run(tmp_path):
                    pytest.fail("second writer acquired lock")
            raise ValueError("interrupted")
    with cache.lock_run(tmp_path):
        pass


def test_plan_force_recomputes_and_resume_revalidates(tmp_path, monkeypatch):
    counts = pd.DataFrame({"n_cells": [1000, 1000]}, index=["A", "B"])
    shared = pd.DataFrame({"island_1": [100., 100.]}, index=["A", "B"])
    candidate = {"lineages": [{"name": label, "coarse_labels": [label], "zoom": True}
                              for label in counts.index], "confirm_shared_islands": True}
    calls = []

    async def fake_agent(*args):
        calls.append(True)
        return plan_module.validate_plan(candidate, list(counts.index), counts, 800, shared)[1]

    monkeypatch.setattr(plan_module, "_run", fake_agent)
    monkeypatch.setattr(plan_module, "lineage_evidence", lambda *a: (counts, None, None, shared))
    args = (None, "coarse", "batch", tmp_path)
    first = plan_module.plan_lineages(*args, model="test")
    assert plan_module.plan_lineages(*args, model="test") == first
    assert len(calls) == 1
    counts.loc["A", "n_cells"] = 1200
    with pytest.raises(ValueError, match="counts or zoom"):
        plan_module.plan_lineages(*args, model="test")
    updated = plan_module.plan_lineages(*args, model="test", force=True)
    assert len(calls) == 2 and updated["lineages"][0]["n_cells"] == 1200


def test_lineage_completion_is_invalidated_before_failed_rerun(tmp_path, monkeypatch):
    sub = AnnData(np.ones((2, 2), dtype="float32"),
                  obs=pd.DataFrame(index=["001", "NA"]))
    sub.obs["msp_ann_cluster"] = "0"
    sub.obs["msp_ann_coarse"] = "A"
    sub.obs["msp_ann_fine"] = "type A"
    for key in lineage.PREVIOUS_COLS:
        sub.obs[key + lineage.PREV_SUFFIX] = "A"
    (tmp_path / "zmip_plan.json").write_text("{}")
    (tmp_path / "lineage_markers.csv").write_text("lineage,gene\n")
    monkeypatch.setattr(lineage, "integrate_adata", lambda *a, **k: None)
    monkeypatch.setattr(lineage, "score_foreign", lambda *a: [])
    monkeypatch.setattr(lineage, "save_single_umap", lambda *a, **k: None)

    def fake_annotate(ad, outdir, *args, **kwargs):
        root = Path(outdir)
        ad.write_h5ad(root / "annotated.h5ad")
        (root / "annotation_proposal.json").write_text("{}")
        (root / "report.html").write_text("<html>complete</html>")
        pd.DataFrame(columns=["cell", "lineage", "cluster"]).to_csv(root / "annotation_removed.csv", index=False)
        pd.DataFrame(columns=["cell", "lineage", "cluster", "reassign_to", "fine_label"]).to_csv(
            root / "annotation_reassigned.csv", index=False)

    monkeypatch.setattr(lineage, "annotate_lineage", fake_annotate)
    args = (sub, "A", ["A"], ["A"], {}, str(tmp_path))
    kwargs = dict(batch_col="sample", species=None, h5ad_path="input.h5ad", resolutions=[1., 2.],
                  n_top_genes=2, n_pcs=1, n_neighbors=1, harmony_kwargs={}, keys_for_foreign=[],
                  language="English", model="test", effort=None, max_turns=1)
    lineage.run_lineage(*args, **kwargs)
    root = tmp_path / "A"
    assert lineage.contract_done(root)
    (tmp_path / "lineage_markers.csv").write_text("lineage,gene\nA,G1\n")
    assert not lineage.contract_done(root)
    (tmp_path / "lineage_markers.csv").write_text("lineage,gene\n")
    assert lineage.contract_done(root)

    def interrupted(*args, **kwargs):
        raise RuntimeError("integration interrupted")

    monkeypatch.setattr(lineage, "integrate_adata", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        lineage.run_lineage(*args, **kwargs)
    # Every old public file is still present, but none can prove completion.
    assert all((root / name).exists() for name in lineage.CONTRACT_FILES)
    assert not lineage.contract_done(root)


@pytest.mark.parametrize("failure", [OSError("subset write failed"), KeyboardInterrupt(), "SIGTERM"])
def test_pool_cleans_launched_process_on_parent_failure(tmp_path, monkeypatch, failure):
    real_popen = subprocess.Popen
    processes = []
    writes = []
    original_term = signal.getsignal(signal.SIGTERM)

    def spawn(cmd, **kwargs):
        proc = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(proc)
        return proc

    def write(path):
        writes.append(path)
        if len(writes) == 2:
            if failure == "SIGTERM":
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            raise failure

    monkeypatch.setattr(lineage.subprocess, "Popen", spawn)
    monkeypatch.setattr(lineage, "subset_for", lambda *a: SimpleNamespace(write_h5ad=write))
    monkeypatch.setattr(lineage, "plan_concurrency", lambda _: (2, 10**12, 1))
    todo = [{"name": n, "coarse_labels": [n], "n_cells": 10} for n in ("A", "B")]
    try:
        with pytest.raises(SystemExit if failure == "SIGTERM" else type(failure)):
            lineage.run_lineages_parallel(None, todo, {"A", "B"}, str(tmp_path), [],
                                          coarse_col="coarse", fine_col="fine")
        assert len(processes) == 1
        assert processes[0].poll() is not None and processes[0].stdout.closed
        assert signal.getsignal(signal.SIGTERM) == original_term
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


@pytest.mark.parametrize("first_exit", [0, 1])
def test_pool_reaps_logs_and_allows_independent_lineage_to_finish(tmp_path, monkeypatch, caplog, first_exit):
    real_popen, real_sleep = subprocess.Popen, time.sleep
    processes = []

    def spawn(cmd, **kwargs):
        code = first_exit if not processes else 0
        proc = real_popen([sys.executable, "-c", f"print('child output'); raise SystemExit({code})"], **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(lineage.subprocess, "Popen", spawn)
    monkeypatch.setattr(lineage.time, "sleep", lambda _: real_sleep(0.01))
    monkeypatch.setattr(lineage, "subset_for", lambda *a: SimpleNamespace(write_h5ad=lambda p: None))
    monkeypatch.setattr(lineage, "plan_concurrency", lambda _: (2, 10**12, 1))
    monkeypatch.setattr(lineage, "contract_done", lambda _: True)
    monkeypatch.setattr(lineage, "load_result", lambda d: {"dir": d})
    todo = [{"name": n, "coarse_labels": [n], "n_cells": 10} for n in ("A", "B")]

    def run():
        return lineage.run_lineages_parallel(None, todo, {"A", "B"}, str(tmp_path), [],
                                             coarse_col="coarse", fine_col="fine")

    with caplog.at_level(logging.INFO, logger="zmip"):
        if first_exit:
            with pytest.raises(RuntimeError, match="failed"):
                run()
        else:
            assert set(run()) == {"A", "B"}
    assert len(processes) == 2 and all(p.poll() is not None and p.stdout.closed for p in processes)
    output = caplog.text
    assert "[A] child output" in output and "[B] child output" in output
    assert "[B] lineage done" in output


def test_check_deg_accepts_subcluster_ids_and_reports_ambiguous_pools(tmp_path, monkeypatch):
    import harness_bridge

    labels = ["0", "5", "1", "5,0", "5,1"]
    ad = AnnData(np.log1p(np.random.default_rng(0).poisson(3, (20, 4))).astype("float32"),
                 obs=pd.DataFrame({annotate.BASE_KEY: pd.Categorical(np.repeat(labels, 4))},
                                  index=[f"c{i}" for i in range(20)]))
    ad.raw = ad
    caches = []
    real_cache = annotate.DegCache

    def capture_cache(*args, **kwargs):
        result = real_cache(*args, **kwargs)
        caches.append(result)
        return result

    async def fake_agent(**kwargs):
        tool = next(t for t in kwargs["tools"] if t.name == "check_deg")
        for reference in ("5,1", '"5,0","5,1"'):
            result = await tool.handler({"cluster": "0", "reference": reference})
            assert not result.get("is_error"), result
        for reference in ("5,0,5,1", "unknown"):
            result = await tool.handler({"cluster": "0", "reference": reference})
            assert result.get("is_error"), result
        return SimpleNamespace(submitted={}, transcript_text="")

    monkeypatch.setattr(annotate, "DegCache", capture_cache)
    monkeypatch.setattr(harness_bridge, "run_agent", fake_agent)
    monkeypatch.setattr(annotate, "_system_prompt", lambda *a: "test")
    asyncio.run(annotate._run_agent(ad, str(tmp_path), "A", ["A"], ["B"], "sample", None, [], {},
                                    np.zeros(20, dtype=bool), [], [], "English", "test", None, 1))
    assert (annotate.BASE_KEY, "0", ("5,1",)) in caches[0]._memo
    assert (annotate.BASE_KEY, "0", ("5,0", "5,1")) in caches[0]._memo


def test_graph_only_paga_matches_full_object_without_mutation(tmp_path, monkeypatch):
    n = 12
    ad = AnnData(np.ones((n, 3), dtype="float32"),
                 obs=pd.DataFrame({"coarse": pd.Categorical(["A"] * 6 + ["B"] * 6), "sample": "s"},
                                  index=[f"c{i}" for i in range(n)]))
    graph = sparse.csr_matrix(np.ones((n, n)) - np.eye(n))
    ad.obsp["connectivities"] = graph
    ad.obsp["distances"] = graph.copy()
    ad.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                           "params": {"n_neighbors": 5}}
    ad.layers["counts"] = ad.X.copy()
    ad.raw = ad
    original_obs, original_uns = ad.obs.copy(), copy.deepcopy(ad.uns)
    full = ad.copy()
    sc.tl.paga(full, groups="coarse")
    expected = full.uns["paga"]["connectivities"].toarray().round(3)
    monkeypatch.setattr(plan_module, "save_single_umap", lambda *a, **k: None)
    _, _, paga, _ = plan_module.lineage_evidence(ad, "coarse", "sample", str(tmp_path))
    assert paga is not None
    np.testing.assert_array_equal(paga.loc[["A", "B"], ["A", "B"]].values, expected)
    pd.testing.assert_frame_equal(ad.obs, original_obs)
    assert ad.uns == original_uns
    np.testing.assert_array_equal(ad.layers["counts"], ad.X)


@pytest.mark.parametrize("item", ["theta", "=1", "theta="])
def test_harmony_errors_are_explicit(item):
    with pytest.raises(ValueError, match="--harmony"):
        parse_harmony([item])


def test_harmony_retains_scalar_and_list_conversion():
    assert parse_harmony(["theta=2,3.5", "device=cpu"]) == {"theta": [2, 3.5], "device": "cpu"}


def test_cli_resume_reruns_only_damaged_lineage_and_force_replans(tmp_path, monkeypatch):
    foreign = importlib.import_module("zmip.foreign")
    merge = importlib.import_module("zmip.merge")
    report = importlib.import_module("zmip.report")
    root, source = tmp_path / "out", tmp_path / "input.h5ad"
    ad = AnnData(np.arange(12, dtype="float32").reshape(6, 2), obs=pd.DataFrame({
        "msp_ann_coarse": pd.Categorical(["A"] * 3 + ["B"] * 3),
        "msp_ann_fine": pd.Categorical(["old A"] * 3 + ["old B"] * 3), "sample": "s",
    }, index=["001", "002", "NA", "004", "005", "006"]))
    ad.layers["counts"] = ad.X.copy()
    ad.uns["msp"] = {"batch_col": "sample"}
    ad.write_h5ad(source)
    counts = pd.DataFrame({"n_cells": [3, 3]}, index=["A", "B"])
    calls = {"plan": 0, "markers": 0, "lineages": []}

    async def fake_agent(*args):
        calls["plan"] += 1
        candidate = {"lineages": [{"name": n, "coarse_labels": [n], "zoom": True} for n in ("A", "B")]}
        return plan_module.validate_plan(candidate, ["A", "B"], counts, 1, None)[1]

    def fake_markers(data, column, outdir):
        calls["markers"] += 1
        pd.DataFrame(columns=foreign.MARKER_COLUMNS).to_csv(Path(outdir) / "lineage_markers.csv", index=False)
        return {"A": [], "B": []}

    def fake_annotate(data, outdir, name, *args, **kwargs):
        calls["lineages"].append(name)
        data.obs["msp_ann_cluster"] = "0"
        data.obs["msp_ann_fine"] = "new " + name
        data.write_h5ad(Path(outdir) / "annotated.h5ad")
        (Path(outdir) / "annotation_proposal.json").write_text(json.dumps({"lineage": name}))
        (Path(outdir) / "report.html").write_text("<html>complete</html>")
        pd.DataFrame(columns=["cell", "lineage", "cluster"]).to_csv(
            Path(outdir) / "annotation_removed.csv", index=False)
        pd.DataFrame(columns=["cell", "lineage", "cluster", "reassign_to", "fine_label"]).to_csv(
            Path(outdir) / "annotation_reassigned.csv", index=False)

    monkeypatch.setenv("ZMIP_PARALLEL", "1")
    monkeypatch.setattr(plan_module, "_run", fake_agent)
    monkeypatch.setattr(plan_module, "lineage_evidence", lambda *a: (counts, None, None, None))
    monkeypatch.setattr(foreign, "lineage_markers", fake_markers)
    monkeypatch.setattr(lineage, "integrate_adata", lambda *a, **k: None)
    monkeypatch.setattr(lineage, "score_foreign", lambda *a: [])
    monkeypatch.setattr(lineage, "save_single_umap", lambda *a, **k: None)
    monkeypatch.setattr(lineage, "annotate_lineage", fake_annotate)
    monkeypatch.setattr(merge, "_figures", lambda *a: None)
    monkeypatch.setattr(report, "generate_report", lambda *a, out_html=None, **k: Path(out_html).write_text("<html>test report</html>"))
    argv = ["zmip", str(source), "--outdir", str(root), "--min-cells", "1"]

    def run(force=False):
        monkeypatch.setattr(sys, "argv", argv + (["--force"] if force else []))
        runpy.run_module("zmip", run_name="__main__")

    run()
    run()
    assert calls == {"plan": 1, "markers": 1, "lineages": ["A", "B"]}
    # A different turn budget, model or report language resumes without recomputing anything.
    monkeypatch.setattr(sys, "argv", argv + ["--max-turns", "5", "--language", "Chinese", "--model", "other"])
    runpy.run_module("zmip", run_name="__main__")
    assert calls == {"plan": 1, "markers": 1, "lineages": ["A", "B"]}
    assert json.loads((root / ".zmip-run.json").read_text())["agent"]["max_turns"] == 5
    (root / "A" / "annotation_removed.csv").write_text("damaged\n")
    run()
    assert calls == {"plan": 1, "markers": 1, "lineages": ["A", "B", "A"]}
    run(force=True)
    assert calls == {"plan": 2, "markers": 2, "lineages": ["A", "B", "A", "A", "B"]}
    result = sc.read_h5ad(root / "annotated_zmip.h5ad")
    assert result.obs_names.tolist() == ad.obs_names.tolist()
    assert result.obs.zmip_ann_fine.tolist() == ["new A"] * 3 + ["new B"] * 3
    np.testing.assert_array_equal(result.layers["counts"], ad.layers["counts"])
