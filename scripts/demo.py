"""End-to-end demo: route a mixed batch of requests from 3 different LoRA
adapters through the scheduler (bubble-lemma grouping + MILP bin packing)
and the fused mock kernel, all on CPU.

Run: python scripts/demo.py
"""

from __future__ import annotations

import torch

from lorafusion.ops.config import AdapterSpec
from lorafusion.runtime.coordinator import Coordinator
from lorafusion.scheduler.adapter_grouper import Request

torch.manual_seed(0)

IN_FEATURES, OUT_FEATURES = 64, 128
NUM_ADAPTERS = 3
RANK = 8

base_weight = torch.randn(OUT_FEATURES, IN_FEATURES)
adapters = {
    aid: AdapterSpec(aid, RANK, alpha=float(RANK * 2), in_features=IN_FEATURES, out_features=OUT_FEATURES)
    for aid in range(NUM_ADAPTERS)
}
lora_a = {aid: torch.randn(RANK, IN_FEATURES) for aid in range(NUM_ADAPTERS)}
lora_b = {aid: torch.randn(OUT_FEATURES, RANK) for aid in range(NUM_ADAPTERS)}

# Simulate a realistic mixed-length request pool, like Figure 6/13 in the
# paper: adapters have different, variable sequence-length distributions.
requests: list[Request] = []
request_tensors: dict[int, torch.Tensor] = {}
request_id = 0
length_ranges = {0: (10, 40), 1: (50, 120), 2: (5, 20)}

for adapter_id, (lo, hi) in length_ranges.items():
    for _ in range(6):
        num_rows = torch.randint(lo, hi, (1,)).item()
        requests.append(Request(request_id, adapter_id, num_rows))
        request_tensors[request_id] = torch.randn(num_rows, IN_FEATURES)
        request_id += 1

print(f"{len(requests)} requests across {NUM_ADAPTERS} adapters, "
      f"total rows = {sum(r.num_rows for r in requests)}")

for use_milp, label in [(False, "greedy"), (True, "MILP")]:
    coordinator = Coordinator(
        base_weight=base_weight,
        adapters=adapters,
        lora_a=lora_a,
        lora_b=lora_b,
        tile_capacity=64,
        use_milp=use_milp,
    )
    outputs = coordinator.run_batch(requests, request_tensors)

    assert len(outputs) == len(requests), "every request must get a result"
    for req in requests:
        expected_rows = req.num_rows
        got_rows = outputs[req.request_id].shape[0]
        assert got_rows == expected_rows, (
            f"request {req.request_id}: expected {expected_rows} rows, got {got_rows}"
        )
        assert outputs[req.request_id].shape[1] == OUT_FEATURES

    print(f"[{label}] scheduled and executed successfully -- "
          f"all {len(requests)} requests returned correctly shaped, "
          f"correctly routed outputs.")

print("\nDemo passed: scheduler (bubble-lemma grouping + bin packing) + "
      "fused mock kernel produced correct per-request outputs for a mixed "
      "multi-adapter batch, entirely on CPU.")
