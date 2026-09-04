# lorafusion

Multi-adapter LoRA serving: fused GPU kernels, a bin-packing scheduler, and a
routing runtime for batching requests across many LoRA adapters sharing one
base model.

## Why

Serving many fine-tuned LoRA adapters behind one base model is bottlenecked
by launch overhead and poor GPU utilization when adapters are run one at a
time. `lorafusion` fuses requests for *different* adapters into a single
kernel launch, using tile-level routing so each row of the batch is scaled
by its own adapter's `A`/`B` matrices.

## Structure

- [`lorafusion/ops/`](lorafusion/ops/) — fused LoRA kernels. `mock_ops.py` is
  a pure PyTorch CPU reference (runs anywhere, including CI and Apple
  Silicon); `triton_ops.py` is the CUDA/Triton implementation, numerically
  validated against the mock on GPU.
- [`lorafusion/scheduler/`](lorafusion/scheduler/) — turns a pool of
  per-adapter requests into fusable tiles: `adapter_grouper.py` groups
  requests by adapter while enforcing the bubble lemma (no request's rows
  may be split across a tile boundary in a way that corrupts routing),
  `milp_packer.py` bin-packs adapter row-counts into tiles (exact via PuLP/
  CBC MILP, with a greedy first-fit-decreasing fallback), and
  `data_batcher.py` assembles/scatters the actual tensors.
- [`lorafusion/runtime/coordinator.py`](lorafusion/runtime/coordinator.py) —
  ties grouping, packing, and kernel execution together into one
  request-batch-in, results-out API.

## Development

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All tests run against `mock_ops` on CPU, so no GPU is required for local
development or CI. GPU kernel correctness is verified separately in
[`colab_verification/run_triton_tests.ipynb`](colab_verification/run_triton_tests.ipynb)
on a Colab GPU runtime.

## Try it

```bash
pip install -r requirements.txt
python scripts/demo.py     # end-to-end: schedule + fuse a mixed multi-adapter batch, CPU-only
pytest tests/ -v            # 12 pass on CPU; 2 GPU-only tests auto-skip
```

## Status

All four layers (kernel math, scheduler, routing runtime, GPU kernel) are
implemented and cross-checked:

- **Kernel math** (`mock_ops.py`): forward + backward verified against
  autograd, CPU-only, always runs.
- **Scheduler** (`adapter_grouper.py`, `milp_packer.py`, `data_batcher.py`):
  bubble-lemma grouping and MILP/greedy bin-packing verified, CPU-only.
- **Runtime** (`coordinator.py`): ties scheduler + kernel together;
  exercised end-to-end by `scripts/demo.py`.
- **GPU kernel** (`triton_ops.py`): a real fused Triton kernel (forward +
  backward), written locally but only *compilable and testable* on a CUDA
  GPU. Verify it by running `colab_verification/run_triton_tests.ipynb` on
  a free Colab GPU runtime -- it runs the same `tests/test_triton_ops.py`
  suite against the CPU mock, plus a throughput benchmark.
