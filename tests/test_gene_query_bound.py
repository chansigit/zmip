import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
from anndata import AnnData

from zmip import annotate


def test_real_gene_handler_bounds_and_cluster_values(tmp_path, monkeypatch):
    import harness_bridge

    genes = [f"G{i:03}" for i in range(40)]
    labels = [str(i) for i in range(77)]
    ad = AnnData(
        np.repeat(np.arange(1, 78, dtype=float)[:, None], 40, axis=1),
        obs=pd.DataFrame({annotate.BASE_KEY: pd.Categorical(labels)}, index=[f"c{i}" for i in labels]),
        var=pd.DataFrame(index=genes),
    )
    original = ad.obs.copy(deep=True)

    async def agent(**kwargs):
        tools = {t.name: t for t in kwargs["tools"]}
        tool = tools["check_genes"]
        result = await tool.handler({"genes": genes, "clusters": []})
        assert result.get("is_error") and "no expression rows" in result["content"][0]["text"]
        for selected in (["not-real" * 10000], "0"):
            bad = await tool.handler({"genes": genes, "clusters": selected})
            assert bad.get("is_error") and len(bad["content"][0]["text"].encode()) < 16384
        good = await tool.handler({"genes": ["G000"], "clusters": ["1", "3"]})
        text = good["content"][0]["text"]
        assert not good.get("is_error")
        assert "2.00|100%" in text and "4.00|100%" in text and "1.00|100%" not in text
        assert len(text.encode()) < 16384
        default = await tool.handler({"genes": ["G000"]})
        assert not default.get("is_error") and "77.00|100%" in default["content"][0]["text"]
        return SimpleNamespace(submitted={}, transcript_text="")

    monkeypatch.setattr(harness_bridge, "run_agent", agent)
    monkeypatch.setattr(annotate, "_system_prompt", lambda *a: "test")
    asyncio.run(
        annotate._run_agent(
            ad,
            str(tmp_path),
            "A",
            ["A"],
            ["B"],
            "sample",
            None,
            [],
            {},
            np.zeros(77, bool),
            [],
            [],
            "English",
            "test",
            None,
            1,
        )
    )
    pd.testing.assert_frame_equal(ad.obs, original)
