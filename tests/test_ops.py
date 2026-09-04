"""Numerical precision tests: fused mock kernels vs. standard unfused PyTorch."""

from __future__ import annotations

import torch

from lorafusion.ops.config import AdapterSpec, TileRoute, TileRoutingConfig
from lorafusion.ops.mock_ops import fused_lora_backward, fused_lora_forward

torch.manual_seed(0)


def _make_problem(
    num_rows_per_adapter: list[int], in_features: int, out_features: int, rank: int
):
    adapters = {}
    lora_a = {}
    lora_b = {}
    routes = []
    cursor = 0

    for adapter_id, num_rows in enumerate(num_rows_per_adapter):
        adapters[adapter_id] = AdapterSpec(
            adapter_id=adapter_id,
            rank=rank,
            alpha=float(rank * 2),
            in_features=in_features,
            out_features=out_features,
        )
        lora_a[adapter_id] = torch.randn(rank, in_features, requires_grad=True)
        lora_b[adapter_id] = torch.randn(out_features, rank, requires_grad=True)
        routes.append(TileRoute(cursor, cursor + num_rows, adapter_id))
        cursor += num_rows

    base_weight = torch.randn(out_features, in_features)
    x = torch.randn(cursor, in_features, requires_grad=True)
    routing = TileRoutingConfig(tile_size=64, routes=routes)
    return x, base_weight, lora_a, lora_b, adapters, routing


def _reference_forward(x, base_weight, lora_a, lora_b, adapters, routing):
    out = torch.zeros(x.shape[0], base_weight.shape[0])
    for route in routing.routes:
        adapter = adapters[route.adapter_id]
        x_slice = x[route.row_start : route.row_end]
        base_out = x_slice @ base_weight.t()
        delta = adapter.scaling * (
            (x_slice @ lora_a[route.adapter_id].t()) @ lora_b[route.adapter_id].t()
        )
        out[route.row_start : route.row_end] = base_out + delta
    return out


def test_forward_matches_reference():
    x, base_weight, lora_a, lora_b, adapters, routing = _make_problem(
        [16, 32, 8], in_features=64, out_features=128, rank=8
    )

    fused_out = fused_lora_forward(x, base_weight, lora_a, lora_b, adapters, routing)
    ref_out = _reference_forward(x, base_weight, lora_a, lora_b, adapters, routing)

    torch.testing.assert_close(fused_out, ref_out, rtol=1e-4, atol=1e-5)


def test_backward_matches_autograd():
    x, base_weight, lora_a, lora_b, adapters, routing = _make_problem(
        [16, 32], in_features=32, out_features=48, rank=4
    )

    ref_out = _reference_forward(x, base_weight, lora_a, lora_b, adapters, routing)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)

    grad_x, grad_a, grad_b = fused_lora_backward(
        grad_out.detach(), x.detach(), base_weight, lora_a, lora_b, adapters, routing
    )

    torch.testing.assert_close(grad_x, x.grad, rtol=1e-4, atol=1e-5)
    for adapter_id in adapters:
        torch.testing.assert_close(
            grad_a[adapter_id], lora_a[adapter_id].grad, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            grad_b[adapter_id], lora_b[adapter_id].grad, rtol=1e-4, atol=1e-5
        )


def test_routing_validate_rejects_gap():
    routing = TileRoutingConfig(
        tile_size=64, routes=[TileRoute(0, 10, 0), TileRoute(15, 20, 1)]
    )
    try:
        routing.validate(20)
        assert False, "expected ValueError for row gap"
    except ValueError:
        pass


def test_routing_validate_rejects_undercoverage():
    routing = TileRoutingConfig(tile_size=64, routes=[TileRoute(0, 10, 0)])
    try:
        routing.validate(20)
        assert False, "expected ValueError for undercoverage"
    except ValueError:
        pass
