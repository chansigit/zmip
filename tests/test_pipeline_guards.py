"""Pipeline boundaries: complete merges, fresh clusterings and tiny lineages."""

import copy
import importlib
import json
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

from zmip.foreign import MARKER_COLUMNS, lineage_markers
from zmip.lineage import load_result, validate_resolutions

merge = importlib.import_module("zmip.merge")
lineage = importlib.import_module("zmip.lineage")
foreign = importlib.import_module("zmip.foreign")
plan_module = importlib.import_module("zmip.plan")


@pytest.fixture
def merge_case(tmp_path, monkeypatch):
    obs = pd.DataFrame({"msp_ann_coarse": pd.Categorical(["A", "A", "B"]),
                        "msp_ann_fine": pd.Categorical(["old A", "old A", "old B"])},
                       index=["001", "002", "NA"])
    ad = AnnData(np.arange(6, dtype="float32").reshape(3, 2), obs=obs)
    ad.layers["counts"] = ad.X.copy()
    ad.raw = ad
    ad.uns["audit"] = "original"
    plan = {"lineages": [{"name": "A", "coarse_labels": ["A"], "zoom": True},
                          {"name": "B", "coarse_labels": ["B"], "zoom": False}]}
    survivors = pd.DataFrame({"msp_ann_cluster": ["0"], "msp_ann_coarse": ["B"],
                               "msp_ann_fine": ["new B"], "zmip_reassigned_to": ["B"]}, index=["001"])
    removed = pd.DataFrame({"cell": ["002"], "lineage": ["A"], "cluster": ["1"],
                            "preannotation": [False], "annotate_remove": [True], "remove_reason": ["doublet"]})
    reassigned = pd.DataFrame({"cell": ["001"], "lineage": ["A"], "cluster": ["0"],
                               "reassign_to": ["B"], "fine_label": ["new B"]})
    monkeypatch.setattr(merge, "_figures", lambda *args: None)
    return ad, plan, survivors, removed, reassigned


def write_result(root, survivors, removed, reassigned):
    root.mkdir(exist_ok=True)
    AnnData(np.ones((len(survivors), 2), dtype="float32"), obs=survivors).write_h5ad(root / "annotated.h5ad")
    removed.to_csv(root / "annotation_removed.csv", index=False)
    reassigned.to_csv(root / "annotation_reassigned.csv", index=False)
    return load_result(root)


def test_merge_preserves_counts_audit_and_reassignment(merge_case, tmp_path):
    ad, plan, survivors, removed, reassigned = merge_case
    original = ad.copy()
    result = write_result(tmp_path / "A", survivors, removed, reassigned)
    kept, rm, ra = merge.merge_back(ad, plan, {"A": result}, tmp_path)
    disk = sc.read_h5ad(tmp_path / "annotated_zmip.h5ad")
    assert kept.obs_names.tolist() == disk.obs_names.tolist() == ["001", "NA"]
    assert disk.obs["msp_ann_coarse"].tolist() == ["A", "B"]
    assert disk.obs["msp_ann_fine"].tolist() == ["old A", "old B"]
    assert disk.obs["zmip_ann_coarse"].tolist() == ["B", "B"]
    assert disk.obs["zmip_ann_fine"].tolist() == ["new B", "old B"]
    assert disk.obs["zmip_lineage"].tolist() == ["B", "B"]
    assert disk.obs.loc["001", "zmip_reassigned_from"] == "A"
    assert disk.obs.loc["001", "zmip_cluster"] == "A:0"
    assert pd.isna(disk.obs.loc["NA", "zmip_cluster"])
    assert disk.uns["audit"] == "original"
    np.testing.assert_array_equal(disk.X, original[["001", "NA"]].X)
    np.testing.assert_array_equal(disk.layers["counts"], original[["001", "NA"]].layers["counts"])
    np.testing.assert_array_equal(disk.raw.X, original[["001", "NA"]].raw.X)
    pd.testing.assert_frame_equal(rm, removed)
    pd.testing.assert_frame_equal(ra, reassigned)


@pytest.mark.parametrize("problem", ["missing", "overlap", "duplicate_kept", "duplicate_removed",
                                      "foreign_kept", "foreign_removed", "duplicate_reassigned",
                                      "removed_reassigned", "wrong_source", "foreign_reassigned"])
def test_invalid_partition_does_not_publish_or_mutate(merge_case, tmp_path, problem):
    ad, plan, survivors, removed, reassigned = merge_case
    if problem == "missing":
        removed = removed.iloc[:0]
    elif problem == "overlap":
        removed.loc[0, "cell"] = "001"
    elif problem == "duplicate_kept":
        survivors = pd.concat([survivors, survivors])
    elif problem == "duplicate_removed":
        removed = pd.concat([removed, removed], ignore_index=True)
    elif problem == "foreign_kept":
        survivors.index = ["NA"]
    elif problem == "foreign_removed":
        removed.loc[0, "cell"] = "NA"
    elif problem == "duplicate_reassigned":
        reassigned = pd.concat([reassigned, reassigned], ignore_index=True)
    elif problem == "removed_reassigned":
        reassigned.loc[0, "cell"] = "002"
    elif problem == "wrong_source":
        removed.loc[0, "lineage"] = "B"
    else:
        reassigned.loc[0, "cell"] = "unknown"
    result = write_result(tmp_path / "A", survivors, removed, reassigned)
    before = ad.obs.copy()
    paths = [tmp_path / f for f in ("annotated_zmip.h5ad", "zmip_removed.csv", "zmip_reassigned.csv")]
    for path in paths:
        path.write_bytes(b"previous completed output")
    with pytest.raises(ValueError, match="inconsistent cell coverage"):
        merge.merge_back(ad, plan, {"A": result}, tmp_path)
    pd.testing.assert_frame_equal(ad.obs, before)
    assert all(path.read_bytes() == b"previous completed output" for path in paths)


@pytest.mark.parametrize("problem", ["missing_result", "extra_result", "duplicate_input",
                                      "duplicate_label", "duplicate_lineage", "unassigned_label"])
def test_merge_rejects_invalid_global_coverage_before_reading(merge_case, tmp_path, monkeypatch, problem):
    ad, plan, *_ = merge_case
    results = {"A": {}}
    if problem == "missing_result":
        results = {}
    elif problem == "extra_result":
        results["B"] = {}
    elif problem == "duplicate_input":
        ad.obs_names = ["001", "001", "NA"]
    elif problem == "duplicate_label":
        plan["lineages"][1]["coarse_labels"] = ["A", "B"]
    elif problem == "duplicate_lineage":
        plan["lineages"][1]["name"] = "A"
    else:
        plan["lineages"][1]["coarse_labels"] = ["C"]
    monkeypatch.setattr(merge.sc, "read_h5ad", lambda *a, **k: pytest.fail("must validate before reading results"))
    with pytest.raises(ValueError):
        merge.merge_back(ad, plan, results, tmp_path)
    assert not (tmp_path / "annotated_zmip.h5ad").exists()


@pytest.mark.parametrize("values", [[], [1.0], [2.0], [0.3], [1., 2., 2.],
                                     [1., 2., 0.], [1., 2., -1.], [1., 2., float("nan")],
                                     [1., 2., float("inf")]])
def test_resolution_preflight_rejects_missing_or_invalid_values(values):
    with pytest.raises(ValueError, match="--resolutions"):
        validate_resolutions(values)


def test_resolution_preflight_preserves_valid_custom_order():
    assert validate_resolutions([2, 0.7, 1]) == (2., 0.7, 1.)


@pytest.mark.parametrize("module", ["zmip", "zmip.lineage"])
def test_cli_rejects_missing_resolution_before_opening_input(module, tmp_path, monkeypatch, capsys):
    outdir = str(tmp_path / "not-created")
    if module == "zmip":
        argv = [module, "missing.h5ad", "--outdir", outdir, "--resolutions", "1.0"]
    else:
        argv = [module, outdir, "A", "--subset", "missing.h5ad", "--h5ad", "missing.h5ad",
                "--batch-col", "sample", "--resolutions", "1.0"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sc, "read_h5ad", lambda *a, **k: pytest.fail("must reject before reading input"))
    with pytest.raises(SystemExit) as exc:
        if module == "zmip":
            runpy.run_module("zmip", run_name="__main__")
        else:
            lineage.main(argv[1:])
    assert exc.value.code == 2
    assert "missing [2.0]" in capsys.readouterr().err
    assert not (tmp_path / "not-created").exists()


def marker_input(groups):
    n = len(groups)
    x = np.zeros((n, 3), dtype="float32")
    for i, group in enumerate(groups):
        x[i, {"A": 0, "B": 1, "tiny": 2}[group]] = np.log1p(10.)
    ad = AnnData(x, obs=pd.DataFrame({"lineage": pd.Categorical(groups)}, index=[f"c{i}" for i in range(n)]))
    ad.layers["counts"] = np.expm1(x)
    ad.raw = ad
    ad.uns["rank_genes_groups"] = {"original": "keep"}
    return ad


def test_singleton_skipped_but_retained_in_reference(tmp_path, monkeypatch, capsys):
    ad = marker_input(["A"] * 4 + ["B"] * 4 + ["tiny"])
    before = ad.copy()
    real_rank = sc.tl.rank_genes_groups
    calls = []

    def record(data, *args, **kwargs):
        calls.append((data.n_obs, kwargs["groups"]))
        return real_rank(data, *args, **kwargs)

    monkeypatch.setattr(sc.tl, "rank_genes_groups", record)
    markers = lineage_markers(ad, "lineage", tmp_path)
    assert calls == [(9, ["A", "B"])]
    assert markers["tiny"] == [] and markers["A"] and markers["B"]
    assert "'tiny': cannot estimate markers" in capsys.readouterr().out
    assert list(pd.read_csv(tmp_path / "lineage_markers.csv")) == MARKER_COLUMNS
    # Eligible results must match Scanpy on the full reference, not a pruned subset.
    expected = before.copy()
    real_rank(expected, "lineage", groups=["A", "B"], method="wilcoxon", use_raw=False, pts=True)
    actual = pd.read_csv(tmp_path / "lineage_markers.csv", dtype={"gene": str})
    for group in ("A", "B"):
        ranked = sc.get.rank_genes_groups_df(expected, group=group)
        ranked = ranked[(ranked.logfoldchanges > 0) & (ranked.pct_nz_reference < foreign.MAX_PCT_OUT)].head(30)
        rows = actual[actual.lineage == group]
        assert rows.gene.tolist() == ranked.names.tolist()
        np.testing.assert_allclose(rows.logFC, ranked.logfoldchanges)
        np.testing.assert_allclose(rows.pct_out, ranked.pct_nz_reference)
    pd.testing.assert_frame_equal(ad.obs, before.obs)
    np.testing.assert_array_equal(ad.X, before.X)
    np.testing.assert_array_equal(ad.layers["counts"], before.layers["counts"])
    assert ad.uns["rank_genes_groups"] == {"original": "keep"}


@pytest.mark.parametrize("groups", [["A"] * 4, ["A", "B", "tiny"], ["A"] * 3 + ["tiny"]])
def test_no_eligible_markers_writes_header_without_running_scanpy(tmp_path, monkeypatch, groups):
    monkeypatch.setattr(sc.tl, "rank_genes_groups", lambda *a, **k: pytest.fail("no eligible comparisons"))
    markers = lineage_markers(marker_input(groups), "lineage", tmp_path)
    assert markers == dict.fromkeys(groups, [])
    table = pd.read_csv(tmp_path / "lineage_markers.csv")
    assert table.empty and list(table) == MARKER_COLUMNS


def test_no_zoom_cli_skips_markers_and_preserves_output_contract(merge_case, tmp_path, monkeypatch):
    ad, plan, *_ = merge_case
    ad.uns["msp"] = {"batch_col": "sample"}
    ad.obs["sample"] = "s"
    for entry in plan["lineages"]:
        entry.update(zoom=False, n_cells=1)
    input_path = tmp_path / "input.h5ad"
    ad.write_h5ad(input_path)
    outdir = tmp_path / "out"
    original = ad.copy()
    def fake_plan(*args, **kwargs):
        (outdir / "zmip_plan.json").write_text(json.dumps(plan))
        return copy.deepcopy(plan)

    monkeypatch.setattr(plan_module, "plan_lineages", fake_plan)
    monkeypatch.setattr(foreign, "lineage_markers", lambda *a, **k: pytest.fail("markers are unnecessary"))
    monkeypatch.setattr(importlib.import_module("zmip.report"), "generate_report", lambda *a: "report.html")
    monkeypatch.setattr(sys, "argv", ["zmip", str(input_path), "--outdir", str(outdir)])
    runpy.run_module("zmip", run_name="__main__")
    output = sc.read_h5ad(outdir / "annotated_zmip.h5ad")
    assert output.obs_names.tolist() == original.obs_names.tolist()
    assert output.obs.zmip_ann_coarse.tolist() == original.obs.msp_ann_coarse.tolist()
    assert output.obs.zmip_ann_fine.tolist() == original.obs.msp_ann_fine.tolist()
    np.testing.assert_array_equal(output.layers["counts"], original.layers["counts"])
    table = pd.read_csv(outdir / "lineage_markers.csv")
    assert table.empty and list(table) == MARKER_COLUMNS
    for filename in ("zmip_removed.csv", "zmip_reassigned.csv"):
        assert pd.read_csv(outdir / filename).empty
