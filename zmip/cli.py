"""Shared integration options for the parent and lineage entry points."""


def add_integration_options(parser):
    parser.add_argument("--resolutions", type=float, nargs="+", default=[0.3, 1.0, 2.0])
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--harmony", action="append", default=[], metavar="KEY=VALUE",
                        help="harmonypy override for the per-lineage re-embedding, repeatable")


def parse_harmony(items):
    def convert(value):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return value

    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--harmony expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key.strip() or not value.strip():
            raise ValueError(f"--harmony expects a nonempty KEY and VALUE, got {item!r}")
        result[key.strip()] = [convert(v) for v in value.split(",")] if "," in value else convert(value)
    return result
