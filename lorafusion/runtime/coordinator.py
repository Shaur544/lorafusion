"""Token-to-adapter routing runtime.

Ties together the scheduler (grouping + packing) and ops (fused kernels)
layers into a single request/response API. Uses the pure-PyTorch mock
kernels by default so it runs anywhere; swap `backend` to use Triton on GPU.
"""

from __future__ import annotations

from types import ModuleType

import torch

from lorafusion.ops import mock_ops
from lorafusion.ops.config import AdapterSpec
from lorafusion.scheduler.adapter_grouper import Request, bucket_by_adapter
from lorafusion.scheduler.data_batcher import build_microbatch, scatter_outputs
from lorafusion.scheduler.milp_packer import greedy_pack, milp_pack


class Coordinator:
    """Routes a batch of per-adapter requests through a fused forward pass."""

    def __init__(
        self,
        base_weight: torch.Tensor,
        adapters: dict[int, AdapterSpec],
        lora_a: dict[int, torch.Tensor],
        lora_b: dict[int, torch.Tensor],
        tile_capacity: int = 128,
        use_milp: bool = False,
        backend: ModuleType = mock_ops,
    ) -> None:
        self.base_weight = base_weight
        self.adapters = adapters
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.tile_capacity = tile_capacity
        self.use_milp = use_milp
        self.backend = backend

    def run_batch(
        self,
        requests: list[Request],
        request_tensors: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Fuse and execute a batch of requests, returning per-request outputs."""
        groups = bucket_by_adapter(requests)
        adapter_rows = {g.adapter_id: g.total_rows for g in groups}

        pack_fn = milp_pack if self.use_milp else greedy_pack
        packing = pack_fn(adapter_rows, self.tile_capacity)

        fused_x, routing, row_order = build_microbatch(
            requests, request_tensors, packing
        )

        fused_out = self.backend.fused_lora_forward(
            fused_x,
            self.base_weight,
            self.lora_a,
            self.lora_b,
            self.adapters,
            routing,
        )

        return scatter_outputs(fused_out, row_order)
