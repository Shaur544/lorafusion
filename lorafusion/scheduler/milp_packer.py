"""Two-stage bin-packing scheduler for fusing multi-adapter requests into tiles.

Stage 1 (optional, exact): a MILP formulation via PuLP that minimizes the
number of tiles used to pack per-adapter row counts, subject to a max tile
capacity. This is exact but can be slow for large numbers of adapters.

Stage 2 (fallback, fast): a first-fit-decreasing greedy bin packer used when
the MILP solver is unavailable, times out, or the problem size is large.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import pulp

    PULP_AVAILABLE = True
except ImportError:  # pragma: no cover
    PULP_AVAILABLE = False


@dataclass(frozen=True)
class PackingResult:
    """A packing plan: list of bins, each a list of (adapter_id, num_rows)."""

    bins: list[list[tuple[int, int]]]
    tile_capacity: int
    method: str  # "milp" or "greedy"

    @property
    def num_bins(self) -> int:
        return len(self.bins)


def greedy_pack(
    adapter_rows: dict[int, int], tile_capacity: int
) -> PackingResult:
    """First-fit-decreasing bin packing over per-adapter row counts.

    Adapters with more than `tile_capacity` rows are split across multiple
    bins so no single bin ever exceeds capacity.
    """
    if tile_capacity <= 0:
        raise ValueError("tile_capacity must be positive")

    items: list[tuple[int, int]] = []
    for adapter_id, rows in adapter_rows.items():
        remaining = rows
        while remaining > 0:
            chunk = min(remaining, tile_capacity)
            items.append((adapter_id, chunk))
            remaining -= chunk

    items.sort(key=lambda item: item[1], reverse=True)

    bins: list[list[tuple[int, int]]] = []
    bin_remaining: list[int] = []

    for adapter_id, chunk in items:
        placed = False
        for i, rem in enumerate(bin_remaining):
            if rem >= chunk:
                bins[i].append((adapter_id, chunk))
                bin_remaining[i] -= chunk
                placed = True
                break
        if not placed:
            bins.append([(adapter_id, chunk)])
            bin_remaining.append(tile_capacity - chunk)

    return PackingResult(bins=bins, tile_capacity=tile_capacity, method="greedy")


def milp_pack(
    adapter_rows: dict[int, int],
    tile_capacity: int,
    max_bins: int | None = None,
    time_limit_s: float = 10.0,
) -> PackingResult:
    """Exact bin packing via MILP (PuLP/CBC). Falls back to greedy on failure.

    Adapters are pre-split into `tile_capacity`-sized chunks (same as
    `greedy_pack`) so the MILP only needs to decide chunk-to-bin assignment,
    keeping the variable count manageable.
    """
    if not PULP_AVAILABLE:
        return greedy_pack(adapter_rows, tile_capacity)

    if tile_capacity <= 0:
        raise ValueError("tile_capacity must be positive")

    items: list[tuple[int, int]] = []
    for adapter_id, rows in adapter_rows.items():
        remaining = rows
        while remaining > 0:
            chunk = min(remaining, tile_capacity)
            items.append((adapter_id, chunk))
            remaining -= chunk

    n_items = len(items)
    upper_bound = max_bins or n_items
    if n_items == 0:
        return PackingResult(bins=[], tile_capacity=tile_capacity, method="milp")

    prob = pulp.LpProblem("tile_bin_packing", pulp.LpMinimize)

    use_bin = [pulp.LpVariable(f"use_{j}", cat="Binary") for j in range(upper_bound)]
    assign = [
        [pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(upper_bound)]
        for i in range(n_items)
    ]

    prob += pulp.lpSum(use_bin)

    for i in range(n_items):
        prob += pulp.lpSum(assign[i][j] for j in range(upper_bound)) == 1

    for j in range(upper_bound):
        prob += (
            pulp.lpSum(items[i][1] * assign[i][j] for i in range(n_items))
            <= tile_capacity * use_bin[j]
        )

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    status = prob.solve(solver)

    if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
        return greedy_pack(adapter_rows, tile_capacity)
    if pulp.LpStatus[status] == "Not Solved":
        return greedy_pack(adapter_rows, tile_capacity)

    bins: list[list[tuple[int, int]]] = []
    for j in range(upper_bound):
        if pulp.value(use_bin[j]) < 0.5:
            continue
        bin_items = [
            items[i] for i in range(n_items) if pulp.value(assign[i][j]) > 0.5
        ]
        if bin_items:
            bins.append(bin_items)

    return PackingResult(bins=bins, tile_capacity=tile_capacity, method="milp")
