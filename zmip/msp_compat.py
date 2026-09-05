"""MSP 0.3 compatibility boundary for newly public presentation/evidence APIs.

MSP 0.3.1 adds public wrappers. Keep published 0.3.0 usable until the
minimum dependency can move; private fallbacks are confined to this module.
"""

import importlib


def _resolve(public_module, public_name, legacy_module, legacy_name):
    module = importlib.import_module(public_module)
    try:
        return getattr(module, public_name)
    except AttributeError:
        return getattr(importlib.import_module(legacy_module), legacy_name)


components = _resolve("msp.evidence", "components", "msp.annotate", "_components")
plot_annotation = _resolve("msp.evidence", "plot_annotation", "msp.annotate", "_plot")
prior_label_columns = _resolve("msp.evidence", "prior_label_columns", "msp.annotate", "_prior_label_columns")
palette = _resolve("msp.evidence", "palette", "msp.annotate", "_palette")
subcluster_once = _resolve("msp.evidence", "subcluster_once", "msp.inspect", "_subcluster_once")
csv_table = _resolve("msp.report", "csv_table", "msp.report", "_csv_table")
img = _resolve("msp.report", "img", "msp.report", "_img")
