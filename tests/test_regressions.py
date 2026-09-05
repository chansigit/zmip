"""Small host-side regressions; no model calls or integration jobs are needed."""

import importlib

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from zmip.annotate import _validate_cluster, _validate_final
from zmip.lineage import lineage_dir, load_result
from zmip.plan import validate_plan


def make_plan(names=("A", "B")):
    return {
        "lineages": [
            {"name": name, "coarse_labels": [label], "zoom": True}
            for name, label in zip(names, ("A", "B"), strict=False)
        ]
    }


def validate(plan, islands=None):
    counts = pd.DataFrame({"n_cells": [1000, 1000]}, index=["A", "B"])
    return validate_plan(plan, ["A", "B"], counts, 800, islands)


def cluster(cid="0", **updates):
    entry = dict(
        cluster_id=cid,
        coarse_label="A",
        fine_label=f"type {cid}",
        merge_target=None,
        action="keep",
        confidence="high",
        evidence=dict.fromkeys(("distinctness", "markers", "foreign", "merge"), "checked"),
        rationale="checked",
    )
    entry.update(updates)
    return entry


@pytest.mark.parametrize(
    "names", [("T/B", "T B"), ("..", "B"), (".", "B"), ("figures", "B"), (".zmip-publish", "B"), (".hidden", "B")]
)
def test_plan_rejects_unsafe_or_colliding_directories(names):
    problems, normalized = validate(make_plan(names))
    assert problems and normalized is None


def test_directory_preserves_existing_slug_and_rejects_external_symlink(tmp_path):
    assert lineage_dir(str(tmp_path), "T/B") == str(tmp_path / "T_B")
    (tmp_path / "A").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        lineage_dir(str(tmp_path), "A")


@pytest.mark.parametrize(
    "payload", [None, [], 1, {"lineages": [None]}, {"lineages": [{"name": "A", "coarse_labels": [{}]}]}]
)
def test_malformed_plan_returns_validation_problems(payload):
    assert validate(payload)[0]


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, []])
def test_plan_requires_real_booleans(value):
    plan = make_plan()
    plan["lineages"][0]["zoom"] = value
    assert validate(plan)[0]
    plan = make_plan()
    plan["confirm_shared_islands"] = value
    assert validate(plan)[0]


def test_island_defenses_and_real_false_are_preserved():
    shared = pd.DataFrame({"island_1": [100.0, 100.0]}, index=["A", "B"])
    plan = make_plan()
    assert validate(plan, shared)[0]
    plan["confirm_shared_islands"] = True
    problems, normalized = validate(plan, shared)
    assert not problems and normalized["host_warnings"]
    plan["lineages"][0]["zoom"] = False
    assert validate(plan)[1]["lineages"][0]["zoom"] is False
    separate = pd.DataFrame({"island_1": [100.0, 0.0], "island_2": [0.0, 100.0]}, index=["A", "B"])
    pooled = {"lineages": [{"name": "Both", "coarse_labels": ["A", "B"]}]}
    assert validate(pooled, separate)[0]


@pytest.mark.parametrize("payload", [None, [], 1])
def test_malformed_cluster_returns_validation_problems(payload):
    assert _validate_cluster(payload, ["0"], ["A"], {"B"})


@pytest.mark.parametrize("value", [None, "", " ", [], 1])
def test_cluster_requires_nonempty_evidence_and_rationale(value):
    entry = cluster(rationale=value)
    assert _validate_cluster(entry, ["0"], ["A"], {"B"})
    entry = cluster()
    entry["evidence"]["foreign"] = value
    assert _validate_cluster(entry, ["0"], ["A"], {"B"})


def test_reassignment_and_merge_defenses_are_preserved():
    assert not _validate_cluster(cluster(), ["0"], ["A"], {"B"})
    assert _validate_cluster(cluster(coarse_label="B"), ["0"], ["A"], {"B"})
    assert _validate_cluster(cluster(action="reassign", reassign_to=[]), ["0"], ["A"], {"B"})
    reassigned = cluster(action="reassign", coarse_label="B", reassign_to="B")
    assert not _validate_cluster(reassigned, ["0"], ["A"], {"B"})
    entries = {"0": cluster(merge_target="1"), "1": cluster("1", action="remove")}
    assert _validate_final(entries, list(entries))
    entries["1"] = cluster("1", fine_label="type 0")
    assert not _validate_final(entries, list(entries))


def test_finalization_names_stale_merge_reference():
    entries = {cid: cluster(cid) for cid in ("0", "1,0", "1,1")}
    entries["0"]["merge_target"] = "1"
    problems = _validate_final(entries, list(entries))
    assert any("cluster 0" in p and "merge_target '1'" in p and "resubmit" in p for p in problems)


def test_disk_results_preserve_identifiers_and_removal_flags(tmp_path):
    ids = ["002", "NA", "null"]
    removed = pd.DataFrame(
        dict(cell=ids, lineage="NA", cluster="01", preannotation=False, annotate_remove=True, remove_reason="doublet")
    )
    reassigned = pd.DataFrame(dict(cell=ids, lineage="NA", cluster="01", reassign_to="NA", fine_label="null"))
    removed.to_csv(tmp_path / "annotation_removed.csv", index=False)
    reassigned.to_csv(tmp_path / "annotation_reassigned.csv", index=False)
    loaded = load_result(tmp_path)
    pd.testing.assert_frame_equal(loaded["removed"], removed)
    pd.testing.assert_frame_equal(loaded["reassigned"], reassigned)
    assert pd.Index(ids).isin(loaded["removed"]["cell"]).all()


def test_fine_figure_uses_coarse_and_fine_pair(tmp_path, monkeypatch):
    merge = importlib.import_module("zmip.merge")
    obs = pd.DataFrame(
        {
            "zmip_ann_coarse": pd.Categorical(["A", "B"]),
            "zmip_ann_fine": pd.Categorical(["cycling", "cycling"]),
            "zmip_lineage": pd.Categorical(["A", "B"]),
            "zmip_action": pd.Categorical(["keep", "keep"]),
            "zmip_reassigned_from": pd.Categorical([None, None]),
        },
        index=["a", "b"],
    )
    ad = AnnData(np.ones((2, 2), dtype="float32"), obs=obs)
    ad.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0]])
    captured = []

    def capture(data, column, path, **kwargs):
        if column == "_fine_id":
            captured.extend(data.obs[column].tolist())

    monkeypatch.setattr(merge, "palette", lambda *args: None)
    monkeypatch.setattr(merge, "save_single_umap", capture)
    figdir = tmp_path / "figures"
    figdir.mkdir()
    merge._figures(ad, ad.copy(), figdir)
    assert captured == ["1", "2"]
    legend = pd.read_csv(tmp_path / "zmip_fine_legend.csv")
    assert list(legend) == ["id", "zmip_ann_coarse", "zmip_ann_fine", "n_cells"]
    assert legend["id"].tolist() == [1, 2]
