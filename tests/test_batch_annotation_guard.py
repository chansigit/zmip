import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from zmip import annotate, report


def entry(**changes):
    result = dict(
        cluster_id="0",
        coarse_label="A",
        fine_label="A state",
        merge_target=None,
        action="remove",
        remove_reason="batch",
        confidence="low",
        evidence=dict.fromkeys(("distinctness", "markers", "foreign", "merge"), "sample specific"),
        rationale="sample enriched",
    )
    result.update(changes)
    return result


def data():
    return AnnData(
        np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame({annotate.BASE_KEY: pd.Categorical(["0", "0"])}, index=["a", "b"]),
    )


def test_guard_preserves_evidence_and_audits_request():
    original = entry()
    guarded = annotate._guard_batch_action(copy.deepcopy(original))
    assert guarded["action"] == "keep" and guarded["remove_reason"] is None
    assert guarded["requested_action"] == "remove" and guarded["requested_remove_reason"] == "batch"
    assert guarded["review_required"] is True
    for key in ("confidence", "coarse_label", "fine_label", "rationale", "evidence"):
        assert guarded[key] == original[key]
    assert annotate._guard_batch_action(copy.deepcopy(guarded)) == guarded


@pytest.mark.parametrize("reason", ["doublet", "low-quality", "ambient", "stress", "other"])
def test_other_reasons_unchanged(reason):
    original = entry(remove_reason=reason)
    assert annotate._guard_batch_action(copy.deepcopy(original)) == original


def test_raw_apply_rejects_before_mutation():
    ad = data()
    before = ad.obs.copy(deep=True)
    with pytest.raises(ValueError, match="batch-only"):
        annotate._apply(ad, annotate.BASE_KEY, {"clusters": [entry()]}, np.zeros(2, bool), "A")
    pd.testing.assert_frame_equal(ad.obs, before)


def test_guarded_apply_preserves_independent_pre_removed():
    ad = data()
    removed, reassigned = annotate._apply(
        ad, annotate.BASE_KEY, {"clusters": [annotate._guard_batch_action(entry())]}, np.array([True, False]), "A"
    )
    assert list(removed["cell"]) == ["a"]
    assert not removed["annotate_remove"].any()
    assert reassigned.empty
    assert list(ad.obs["msp_ann_action"]) == ["remove", "keep"]


def test_real_submit_finalize_and_lineage_constraint(tmp_path, monkeypatch):
    import harness_bridge

    async def agent(**kwargs):
        handlers = {t.name: t.handler for t in kwargs["tools"]}
        invalid = await handlers["submit_cluster"]({"cluster_json": json.dumps(entry(coarse_label="B"))})
        assert invalid.get("is_error")
        missing = await handlers["finalize_annotation"]({"overall": "test"})
        assert missing.get("is_error")
        accepted = await handlers["submit_cluster"]({"cluster_json": json.dumps(entry())})
        assert not accepted.get("is_error"), accepted
        final = await handlers["finalize_annotation"]({"overall": "test"})
        assert not final.get("is_error"), final
        return SimpleNamespace(submitted={}, transcript_text="")

    monkeypatch.setattr(harness_bridge, "run_agent", agent)
    monkeypatch.setattr(annotate, "_system_prompt", lambda *a: "test")
    asyncio.run(
        annotate._run_agent(
            data(),
            str(tmp_path),
            "A",
            ["A"],
            ["B"],
            "sample",
            None,
            [],
            {},
            np.zeros(2, bool),
            [],
            [],
            "English",
            "test",
            None,
            1,
        )
    )
    saved = json.loads((tmp_path / "annotation_proposal.json").read_text())
    assert saved["clusters"][0]["review_required"] is True
    assert saved["clusters"][0]["action"] == "keep"


def test_normalized_merge_conflict_still_rejected():
    entries = {
        "0": annotate._guard_batch_action(entry(merge_target="1")),
        "1": entry(cluster_id="1", remove_reason="doublet"),
    }
    assert annotate._validate_final(entries, ["0", "1"])


def test_global_report_exposes_host_review(tmp_path):
    target = tmp_path / "A"
    target.mkdir()
    (target / "annotation_proposal.json").write_text(json.dumps({"clusters": [annotate._guard_batch_action(entry())]}))
    text = report._section_lineages(str(tmp_path), {"lineages": [{"name": "A", "zoom": True, "n_cells": 2}]})
    assert "retained for review" in text and "requested reason" in text and "batch" in text
