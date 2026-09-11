"""Re-export point for the MSP presentation/evidence APIs zmip reuses.

Historically a compatibility shim over msp 0.3.0 (which had these seven
helpers as private), before msp 0.3.1 made them public. The minimum msp
dependency (pyproject.toml) is now well past that, so this module is a
plain re-export kept only so annotate.py/report.py/merge.py don't each
import from three different msp submodules.
"""

from msp.evidence import components, palette, plot_annotation, prior_label_columns, subcluster_once
from msp.report import csv_table, img

__all__ = [
    "components",
    "csv_table",
    "img",
    "palette",
    "plot_annotation",
    "prior_label_columns",
    "subcluster_once",
]
