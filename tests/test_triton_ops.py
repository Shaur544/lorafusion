"""GPU-only correctness tests: Triton kernels vs. the CPU mock reference.

Skipped automatically on machines without CUDA (e.g. Apple Silicon). Run for
real on a Colab GPU runtime via `colab_verification/run_triton_tests.ipynb`,
which invokes this same file with `pytest`.
"""

from __future__ import annotations

import pytest
import torch

from lorafusion.ops import mock_ops, triton_ops
from lorafusion.ops.config import AdapterSpec, TileRoute, TileRoutingConfig

pytestmark = pytest.mark.skipif(
    not (triton_ops.TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="requires a CUDA GPU with triton installed",
)


def _assert_close_fp16(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare a fp16-tensor-core kernel's output against a fp32 reference.

    The kernel computes in fp16 (accumulating in fp32), matching the paper's
    own half-precision-training setup, so it will never be bit-exact with a
    pure-fp32 reference. A handful of outlier elements with large *relative*
    error near-zero values is expected fp16 behavior, not a bug -- but the
    bulk of elements should still match tightly, and no element should be
    wildly off (which would indicate a real bug, e.g. an indexing error).
    """
    diff = (actual - expected).abs()
    tight_tol = 5e-3 + 2e-2 * expected.abs()
    frac_outliers = (diff > tight_tol).float().mean().item()
    assert frac_outliers < 0.01, (
        f"{frac_outliers:.1%} of elements exceed the tight fp16 tolerance "
        "(expected < 1%, isolated rounding outliers only)"
    )
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=1e-1)


def _make_problem(num_rows_per_adapter, in_features, out_features, rank, device):
    torch.manual_seed(0)
    adapters, lora_a, lora_b, routes = {}, {}, {}, []
    cursor = 0
    for adapter_id, num_rows in enumerate(num_rows_per_adapter):
        adapters[adapter_id] = AdapterSpec(
            adapter_id=adapter_id,
            rank=rank,
            alpha=float(rank * 2),
            in_features=in_features,
            out_features=out_features,
        )
        lora_a[adapter_id] = torch.randn(rank, in_features, device=device)
        lora_b[adapter_id] = torch.randn(out_features, rank, device=device)
        routes.append(TileRoute(cursor, cursor + num_rows, adapter_id))
        cursor += num_rows

    base_weight = torch.randn(out_features, in_features, device=device)
    x = torch.randn(cursor, in_features, device=device)
    routing = TileRoutingConfig(tile_size=64, routes=routes)
    return x, base_weight, lora_a, lora_b, adapters, routing


def test_triton_forward_matches_mock():
    x, base_weight, lora_a, lora_b, adapters, routing = _make_problem(
        [16, 32, 8], in_features=64, out_features=128, rank=8, device="cuda"
    )

    triton_out = triton_ops.fused_lora_forward(
        x, base_weight, lora_a, lora_b, adapters, routing
    )
    mock_out = mock_ops.fused_lora_forward(
        x.cpu(),
        base_weight.cpu(),
        {k: v.cpu() for k, v in lora_a.items()},
        {k: v.cpu() for k, v in lora_b.items()},
        adapters,
        routing,
    )

    _assert_close_fp16(triton_out.cpu(), mock_out)


def test_triton_backward_matches_mock():
    x, base_weight, lora_a, lora_b, adapters, routing = _make_problem(
        [16, 32], in_features=32, out_features=48, rank=4, device="cuda"
    )
    grad_out = torch.randn(x.shape[0], base_weight.shape[0], device="cuda")

    t_grad_x, t_grad_a, t_grad_b = triton_ops.fused_lora_backward(
        grad_out, x, base_weight, lora_a, lora_b, adapters, routing
    )
    m_grad_x, m_grad_a, m_grad_b = mock_ops.fused_lora_backward(
        grad_out.cpu(),
        x.cpu(),
        base_weight.cpu(),
        {k: v.cpu() for k, v in lora_a.items()},
        {k: v.cpu() for k, v in lora_b.items()},
        adapters,
        routing,
    )

    _assert_close_fp16(t_grad_x.cpu(), m_grad_x)
    for adapter_id in adapters:
        _assert_close_fp16(t_grad_a[adapter_id].cpu(), m_grad_a[adapter_id])
        _assert_close_fp16(t_grad_b[adapter_id].cpu(), m_grad_b[adapter_id])
