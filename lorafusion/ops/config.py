"""Tile-level routing configuration for multi-adapter LoRA fusion kernels.

These dataclasses describe how a fused batch is split into GPU-tile-sized
segments, each segment tagged with the adapter it should route to. Both the
mock (pure PyTorch) and Triton kernels consume the same config so that their
outputs are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AdapterSpec:
    """Static description of a single LoRA adapter."""

    adapter_id: int
    rank: int
    alpha: float
    in_features: int
    out_features: int

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass(frozen=True)
class TileRoute:
    """One contiguous range of rows in the fused batch routed to one adapter."""

    row_start: int
    row_end: int
    adapter_id: int

    @property
    def num_rows(self) -> int:
        return self.row_end - self.row_start


@dataclass(frozen=True)
class TileRoutingConfig:
    """Full routing plan for a fused forward/backward pass."""

    tile_size: int
    routes: list[TileRoute] = field(default_factory=list)

    def num_tiles(self) -> int:
        return sum(
            (route.num_rows + self.tile_size - 1) // self.tile_size
            for route in self.routes
        )

    def validate(self, total_rows: int) -> None:
        covered = 0
        prev_end = 0
        for route in self.routes:
            if route.row_start != prev_end:
                raise ValueError(
                    f"Routing gap/overlap at row {route.row_start}, "
                    f"expected {prev_end}"
                )
            covered += route.num_rows
            prev_end = route.row_end
        if covered != total_rows:
            raise ValueError(
                f"Routing covers {covered} rows, expected {total_rows}"
            )
