import asyncio
import json
from types import SimpleNamespace

import harness_bridge
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from test_annotation_status import body, entry

from zmip import annotate as A


def test_restart_preserves_split_cells_and_dangling_merge_for_correction(tmp_path, monkeypatch):
    obj = AnnData(
        np.ones((4, 2)),
        obs=pd.DataFrame(
            {
                A.BASE_KEY: pd.Categorical(["0", "0", "1", "1"]),
            },
            index=["001", "002", "003", "004"],
        ),
    )
    calls, splits = [], []

    def split(ad, key, cluster, resolution, new_key, mask):
        splits.append(cluster)
        ad.obs[new_key] = pd.Categorical(["0,0", "0,1", "1", "1"])
        return 2, "split"

    async def agent(**kwargs):
        handlers = {t.name: t.handler for t in kwargs["tools"]}
        status = body(await handlers["annotation_status"]({}))
        calls.append(status)
        if len(calls) == 1:
            await handlers["submit_cluster"]({"cluster_json": json.dumps(entry("1", merge_target="0"))})
            await handlers["subcluster"]({"cluster": "0", "resolution": 0.5})
            await handlers["submit_cluster"]({"cluster_json": json.dumps(entry("0,0", merge_target="1"))})
            raise RuntimeError("simulated interruption")
        assert status["cluster_key"] == "zmip_sub1" and status["submitted_count"] == 2
        assert status["pending_ids"] == ["0,1"]
        assert (await handlers["finalize_annotation"]({})).get("is_error")
        for c in ("1", "0,1"):
            await handlers["submit_cluster"](
                {"cluster_json": json.dumps(entry(c, merge_target=None if c == "1" else "1"))}
            )
        result = await handlers["finalize_annotation"]({"overall": "restored"})
        assert not result.get("is_error"), result
        return SimpleNamespace(submitted=result["_submitted"], transcript_text="")

    monkeypatch.setattr(A, "subcluster_once", split)
    monkeypatch.setattr(A, "_system_prompt", lambda *a: "test")
    monkeypatch.setattr(harness_bridge, "run_agent", agent)

    def run(data=None):
        data = obj.copy() if data is None else data
        result = asyncio.run(
            A._run_agent(
                data,
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
        assert data.obs["zmip_sub1"].astype(str).tolist() == ["0,0", "0,1", "1", "1"]
        return result

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run()
    result = run()
    assert run() == result and len(calls) == 2 and splits == ["0"]
    with pytest.raises(ValueError, match="input, evidence"):
        run(obj[[1, 0, 2, 3]].copy())
