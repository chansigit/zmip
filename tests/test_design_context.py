"""--design-context flows from the zmip outdir into every lineage's annotation prompt."""

from msp.report import write_design_context

from zmip import annotate

DESIGN = "sample P1: subtissue=Immune (CD45+), mouse.id=3_8_M"


def prompt(lineage_dir):
    return annotate._system_prompt(
        str(lineage_dir), "Immune", ["T cell"], ["Fibroblast"], ["0", "1"], "sample", "mouse", [], [], "English"
    )


def test_lineage_prompt_inherits_parent_design(tmp_path):
    lineage = tmp_path / "immune"
    lineage.mkdir()
    assert "Study design" not in prompt(lineage)
    write_design_context(str(tmp_path), DESIGN)
    text = prompt(lineage)
    assert "Study design (from the caller — treat as ground truth about how samples were produced):" in text
    assert f"<<<\n{DESIGN}\n>>>" in text
