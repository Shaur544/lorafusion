"""Pure PyTorch CPU reference implementation of fused multi-adapter LoRA.

This module has no dependency on Triton/CUDA so it can run in CI and on
machines without a GPU (e.g. Apple Silicon). It is the numerical ground
truth that `triton_ops.py` is validated against in `tests/test_ops.py`.
"""

from __future__ import annotations

import torch

from lorafusion.ops.config import AdapterSpec, TileRoutingConfig


def fused_lora_forward(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_a: dict[int, torch.Tensor],
    lora_b: dict[int, torch.Tensor],
    adapters: dict[int, AdapterSpec],
    routing: TileRoutingConfig,
) -> torch.Tensor:
    """Compute y = x @ base_weight^T + scaling * (x @ A^T) @ B^T, per-row routed.

    Args:
        x: (num_rows, in_features) fused input batch.
        base_weight: (out_features, in_features) shared frozen base weight.
        lora_a: adapter_id -> (rank, in_features) LoRA A matrix.
        lora_b: adapter_id -> (out_features, rank) LoRA B matrix.
        adapters: adapter_id -> AdapterSpec (for scaling factor).
        routing: contiguous row ranges mapping to adapter ids.

    Returns:
        (num_rows, out_features) output tensor.
    """
    routing.validate(x.shape[0])

    out = x @ base_weight.t()

    for route in routing.routes:
        adapter = adapters[route.adapter_id]
        x_slice = x[route.row_start : route.row_end]
        a = lora_a[route.adapter_id]
        b = lora_b[route.adapter_id]
        delta = (x_slice @ a.t()) @ b.t()
        out[route.row_start : route.row_end] += adapter.scaling * delta

    return out


def fused_lora_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_a: dict[int, torch.Tensor],
    lora_b: dict[int, torch.Tensor],
    adapters: dict[int, AdapterSpec],
    routing: TileRoutingConfig,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Backward pass: returns (grad_x, grad_lora_a, grad_lora_b).

    Base weight is treated as frozen (no grad returned for it), matching
    standard LoRA fine-tuning where only adapter params are trainable.
    """
    routing.validate(x.shape[0])

    grad_x = grad_out @ base_weight
    grad_lora_a: dict[int, torch.Tensor] = {}
    grad_lora_b: dict[int, torch.Tensor] = {}

    for route in routing.routes:
        adapter = adapters[route.adapter_id]
        scaling = adapter.scaling
        x_slice = x[route.row_start : route.row_end]
        g_slice = grad_out[route.row_start : route.row_end]
        a = lora_a[route.adapter_id]
        b = lora_b[route.adapter_id]

        xa = x_slice @ a.t()  # (rows, rank)

        grad_b = scaling * (g_slice.t() @ xa)  # (out_features, rank)
        grad_xa = scaling * (g_slice @ b)  # (rows, rank)
        grad_a = grad_xa.t() @ x_slice  # (rank, in_features)
        grad_x_slice = grad_xa @ a  # (rows, in_features)

        grad_x[route.row_start : route.row_end] += grad_x_slice

        grad_lora_a[route.adapter_id] = (
            grad_lora_a.get(route.adapter_id, 0) + grad_a
        )
        grad_lora_b[route.adapter_id] = (
            grad_lora_b.get(route.adapter_id, 0) + grad_b
        )

    return grad_x, grad_lora_a, grad_lora_b
