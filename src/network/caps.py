"""Hard ceilings for cell-cell communication inference.

These are the algorithm's own defense-in-depth bounds, enforced regardless of
what `CellCommConfig` allows -- `CellCommConfig` values must never exceed these.
They exist so `infer_cell_communication` stays a synchronous, bounded-time
request (no task queue backs this endpoint).
"""

MAX_CELLS = 2000
MAX_GENES = 500
MAX_CLUSTERS = 10
MAX_PERMUTATIONS = 1000
