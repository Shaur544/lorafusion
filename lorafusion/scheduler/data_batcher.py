"""Microbatch data generation and merging for fused multi-adapter execution.

Given adapter-grouped requests and a packing plan, this module produces the
concrete fused input tensor (rows in tile order) and a matching
`TileRoutingConfig`, and reverses the process to scatter fused outputs back
to per-request results.
"""

from __future__ import annotations

import torch

from lorafusion.ops.config import TileRoute, TileRoutingConfig
from lorafusion.scheduler.adapter_grouper import Request
from lorafusion.scheduler.milp_packer import PackingResult


def build_microbatch(
    requests: list[Request],
    request_tensors: dict[int, torch.Tensor],
    packing: PackingResult,
) -> tuple[torch.Tensor, TileRoutingConfig, list[int]]:
    """Assemble one fused input tensor plus routing config from a packing plan.

    Args:
        requests: all requests being batched (for row-count lookup).
        request_tensors: request_id -> (num_rows, in_features) tensor.
        packing: output of `milp_pack`/`greedy_pack`, one bin per tile.

    Returns:
        fused_x: concatenated (total_rows, in_features) tensor, tile-major.
        routing: TileRoutingConfig describing adapter row ranges.
        row_order: request_id repeated per-row, in fused-row order — used by
            `scatter_outputs` to map results back to requests.
    """
    request_by_id = {r.request_id: r for r in requests}
    in_features = next(iter(request_tensors.values())).shape[1]

    chunks: list[torch.Tensor] = []
    routes: list[TileRoute] = []
    row_order: list[int] = []
    cursor = 0

    # Requests can be split across multiple tiles by the packer (an adapter's
    # total rows may exceed one tile's capacity), so we must track how much
    # of each request has already been consumed across ALL tiles -- not just
    # within one -- otherwise earlier rows get duplicated into every tile
    # that adapter appears in and later requests never get scheduled.
    consumed: dict[int, int] = {}
    by_adapter: dict[int, list[Request]] = {}
    for r in requests:
        by_adapter.setdefault(r.adapter_id, []).append(r)

    for tile in packing.bins:
        for adapter_id, num_rows in tile:
            matching = by_adapter.get(adapter_id, [])
            rows_needed = num_rows
            for req in matching:
                if rows_needed <= 0:
                    break
                tensor = request_tensors[req.request_id]
                already = consumed.get(req.request_id, 0)
                remaining_in_req = tensor.shape[0] - already
                if remaining_in_req <= 0:
                    continue
                take = min(rows_needed, remaining_in_req)
                chunks.append(tensor[already : already + take])
                row_order.extend([req.request_id] * take)
                rows_needed -= take
                consumed[req.request_id] = already + take

            routes.append(
                TileRoute(
                    row_start=cursor,
                    row_end=cursor + num_rows,
                    adapter_id=adapter_id,
                )
            )
            cursor += num_rows

    fused_x = (
        torch.cat(chunks, dim=0)
        if chunks
        else torch.empty((0, in_features))
    )
    routing = TileRoutingConfig(tile_size=packing.tile_capacity, routes=routes)
    return fused_x, routing, row_order


def scatter_outputs(
    fused_out: torch.Tensor, row_order: list[int]
) -> dict[int, torch.Tensor]:
    """Split a fused output tensor back into per-request tensors using the
    `row_order` produced by `build_microbatch`.
    """
    results: dict[int, list[torch.Tensor]] = {}
    for row_idx, request_id in enumerate(row_order):
        results.setdefault(request_id, []).append(fused_out[row_idx])

    return {
        request_id: torch.stack(rows, dim=0)
        for request_id, rows in results.items()
    }
