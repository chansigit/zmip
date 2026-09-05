import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from zmip import annotate as A


def entry(cid, **changes):
    result = dict(
        cluster_id=str(cid),
        coarse_label="A",
        fine_label="A state",
        action="keep",
        merge_target=None,
        confidence="high",
        evidence=dict.fromkeys(("distinctness", "markers", "foreign", "merge"), "evidence"),
        rationale="supported",
        remove_reason=None,
        reassign_to=None,
    )
    result.update(changes)
    return result


def body(response):
    text = response["content"][0]["text"]
    assert len(text.encode()) <= 16384
    assert not response.get("is_error"), response
    return json.loads(text)


def test_96_cluster_summary_is_bounded_and_preserves_reassign_review():
    clusters = [str(i) for i in range(96)]
    entries = {c: entry(c, rationale="long evidence" * 5000) for c in clusters[:80]}
    entries["0"].update(action="reassign", coarse_label="B", reassign_to="B", review_required=True)
    pending = []
    accepted = []
    offset = 0
    while True:
        result = body(A._annotation_status(entries, clusters, offset=offset, cluster_key="zmip_sub3", n_sub=3))
        assert result["cluster_key"] == "zmip_sub3" and result["n_sub"] == 3
        assert result["submitted_count"] == 80 and result["pending_count"] == 16
        assert len(result["submitted"]) <= 8 and len(result["pending_ids"]) <= 8
        assert "long evidence" not in json.dumps(result)
        pending += result["pending_ids"]
        accepted += result["submitted"]
        if result["next_offset"] is None:
            break
        offset = result["next_offset"]
    assert pending == clusters[80:]
    assert [x["cluster_id"] for x in accepted] == clusters[:80]
    assert accepted[0]["reassign_to"] == "B" and accepted[0]["review_required"] is True


def test_full_entry_unicode_audit_json_paginates_exactly():
    saved = entry(
        "0,1",
        rationale="中文 evidence" * 2000,
        requested_action="remove",
        requested_remove_reason="other",
        host_adjustment={"policy": "batch_annotation_non_destructive_v1"},
        review_required=True,
    )
    offset = 0
    chunks = []
    while True:
        result = body(
            A._annotation_status(
                {"0,1": saved}, ["0,1"], cluster="0,1", offset=offset, cluster_key="zmip_sub1", n_sub=1
            )
        )
        assert len(result["entry_json"].encode()) <= 6000
        assert result["cluster_key"] == "zmip_sub1"
        chunks.append(result["entry_json"])
        if result["next_offset"] is None:
            break
        offset = result["next_offset"]
    assert json.loads("".join(chunks)) == saved


@pytest.mark.parametrize("cluster,offset", [("missing", 0), ("", -1), ("", True), ("", 999), ("0", 999999)])
def test_invalid_queries_are_bounded_errors(cluster, offset):
    response = A._annotation_status({"0": entry("0")}, ["0"], cluster=cluster, offset=offset)
    assert response.get("is_error") and len(response["content"][0]["text"].encode()) <= 16384


def test_pending_detail_does_not_invent_submission():
    result = body(A._annotation_status({}, ["0,0"], cluster="0,0", cluster_key="zmip_sub1", n_sub=1))
    assert result["submitted"] is False and result["cluster_key"] == "zmip_sub1"


def test_label_preview_and_identifier_overflow_are_bounded():
    result = body(A._annotation_status({"0": entry("0", fine_label="x" * 5000)}, ["0"]))
    assert len(result["submitted"][0]["fine_label"]) == 97
    huge = "x" * 20000
    response = A._annotation_status({huge: entry(huge)}, [huge])
    assert response.get("is_error") and len(response["content"][0]["text"].encode()) <= 16384


def test_real_handlers_follow_subcluster_and_preserve_merge_protocol(tmp_path, monkeypatch):
    import harness_bridge

    ad = AnnData(
        np.ones((4, 2)),
        obs=pd.DataFrame({A.BASE_KEY: pd.Categorical(["0", "0", "1", "1"])}, index=["a", "b", "c", "d"]),
    )

    def split(ad, key, cluster, resolution, new_key, pre_removed):
        assert key == A.BASE_KEY and cluster == "0"
        ad.obs[new_key] = pd.Categorical(["0,0", "0,1", "1", "1"])
        return 2, "split"

    async def agent(**kwargs):
        handlers = {t.name: t.handler for t in kwargs["tools"]}

        async def submit(e):
            result = await handlers["submit_cluster"]({"cluster_json": json.dumps(e)})
            assert not result.get("is_error"), result

        await submit(entry("0"))
        await submit(entry("1", merge_target="0"))
        before = body(await handlers["annotation_status"]({"cluster": "", "offset": 0}))
        assert before["pending_count"] == 0 and before["cluster_key"] == A.BASE_KEY
        await handlers["subcluster"]({"cluster": "0", "resolution": 0.5})
        after = body(await handlers["annotation_status"]({"cluster": "", "offset": 0}))
        assert after["cluster_key"] == "zmip_sub1" and after["n_sub"] == 1
        assert set(after["pending_ids"]) == {"0,0", "0,1"}
        assert after["submitted"][0]["cluster_id"] == "1" and after["submitted"][0]["merge_target_current"] is False
        assert (await handlers["annotation_status"]({"cluster": "0", "offset": 0})).get("is_error")
        assert (await handlers["finalize_annotation"]({"overall": "test"})).get("is_error")
        saved = body(await handlers["annotation_status"]({"cluster": "1", "offset": 0}))
        assert json.loads(saved["entry_json"])["merge_target"] == "0"
        await submit(entry("0,0", merge_target="1"))
        await submit(entry("0,1", merge_target="1"))
        await submit(entry("1"))
        ready = body(await handlers["annotation_status"]({"cluster": "", "offset": 0}))
        assert ready["submitted_count"] == 3 and ready["pending_count"] == 0
        assert not (await handlers["finalize_annotation"]({"overall": "test"})).get("is_error")
        return SimpleNamespace(submitted={}, transcript_text="")

    monkeypatch.setattr(A, "subcluster_once", split)
    monkeypatch.setattr(A, "_system_prompt", lambda *a: "test")
    monkeypatch.setattr(harness_bridge, "run_agent", agent)
    asyncio.run(
        A._run_agent(
            ad,
            str(tmp_path),
            "A",
            ["A"],
            ["B"],
            "sample",
            None,
            [],
            {},
            np.zeros(4, bool),
            [],
            [],
            "English",
            "test",
            None,
            1,
        )
    )
    proposal = json.loads((tmp_path / "annotation_proposal.json").read_text())
    assert {e["cluster_id"] for e in proposal["clusters"]} == {"0,0", "0,1", "1"}
    assert proposal["cluster_key"] == "zmip_sub1"
