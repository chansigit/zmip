"""msp_compat is a plain re-export of msp's public evidence/report API."""

from zmip import msp_compat


def test_reexports_every_helper_from_its_public_msp_module():
    from msp import evidence, report

    for name in ("components", "palette", "plot_annotation", "prior_label_columns", "subcluster_once"):
        assert getattr(msp_compat, name) is getattr(evidence, name)
    for name in ("csv_table", "img"):
        assert getattr(msp_compat, name) is getattr(report, name)
    assert set(msp_compat.__all__) == {
        "components", "csv_table", "img", "palette", "plot_annotation",
        "prior_label_columns", "subcluster_once",
    }
