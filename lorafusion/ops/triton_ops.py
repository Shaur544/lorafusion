"""Triton GPU kernels for fused multi-adapter LoRA.

This module requires a CUDA-capable GPU and `triton` to import successfully.
It mirrors the API of `mock_ops.py` exactly so tests can swap between the two
backends transparently. Compile/run it on Colab via
`colab_verification/run_triton_tests.ipynb`.

Kernel design
-------------
Both the forward delta-add and the backward dX computation have the same
algebraic shape:

    out = P @ M1 + scale * (P @ M2) @ M3

Forward:  P=X,  M1=W^T,  M2=A^T,  M3=B^T   ->  Y  = X@W^T + scale*(X@A^T)@B^T
Backward: P=dY, M1=W,    M2=B,    M3=A     ->  dX = dY@W  + scale*(dY@B)@A

so a single `_fused_double_gemm` kernel serves both, matching FusedLoRA's
Figure 10 fusion of the base GEMM with the LoRA branch. The fusion win is
that `P` is read from DRAM exactly once and feeds *both* `tl.dot`s while it
sits in shared memory, and `out` is written exactly once -- an unfused
implementation reads X twice and does a read-modify-write over the whole
(M,N) output to add the delta.

`dA`/`dB` are left as plain PyTorch matmuls (rank is tiny, so this path is
cheap) -- this mirrors the paper's own design, where the small-tensor
gradient op is explicitly left unfused ("Operation 4 remains unchanged").

Why operands are converted to fp16 *outside* the kernel
-------------------------------------------------------
The compute is fp16-tensor-core with fp32 accumulation. The tempting thing is
to let callers pass fp32 tensors and do `.to(tl.float16)` inside the kernel
just before `tl.dot`. That is a serious performance trap, and was this
kernel's dominant bottleneck:

  * The tile still crosses DRAM and lands in shared memory as *fp32*, so it
    costs 4 bytes/element in both. Shared memory is the binding constraint on
    a tiled GEMM: with fp32 tiles, a 128x128x32 stage needs ~102 KB across 3
    pipeline stages, well over a Turing SM's 64 KB budget. Triton's autotuner
    does not warn about this -- it catches `OutOfResources` and silently
    *prunes* the config. Every large-tile config gets dropped and the kernel
    quietly falls back to 64x64 or 32x32 tiles.
  * Small tiles are catastrophic for a GEMM, because each operand is re-read
    from DRAM once per block along the other axis. At 32x32 the base GEMM
    alone moves 4.3 GB for a 1024x4096x4096 problem -- a ~17 ms DRAM floor on
    a T4 before a single FLOP is counted. In fp16 at 128x128 the same problem
    moves 0.54 GB, a ~2 ms floor.

So the fp32->fp16 conversion happens once, up front, on the host side. This
costs nothing in accuracy relative to converting inside the kernel (the dot
was fp16 either way) and buys back both the bandwidth and the tile size.

Why there is no `tl.trans` any more
-----------------------------------
`base_weight`/`A`/`B` are stored (out,in), (rank,in), (out,rank), but the
forward algebra wants their transposes. Reading them transposed via raw
strides puts the fast-varying tile dimension on a large stride, which is an
uncoalesced access pattern. The previous fix for that was to load each tile
in its natural orientation and flip it in-register with `tl.trans`, but on
fp32 data that is a 4-byte shared-memory transpose: heavy bank conflicts, and
it doubles shared memory usage again on top of the problem above.

These three tensors are *static* across calls -- `base_weight` is frozen by
construction, and A/B change only on an optimizer step. So they are
transposed and cast to fp16 exactly once and memoized (see `_prepare`,
invalidated on tensor identity, storage pointer, and version counter). The
kernel then has a single, clean, fully-coalesced NN path with no transpose
branch at all, for both forward and backward.
"""

from __future__ import annotations

import weakref

import torch

from lorafusion.ops.config import AdapterSpec, TileRoutingConfig

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only off-GPU
    TRITON_AVAILABLE = False


def _require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "triton_ops requires a CUDA GPU with `triton` installed. "
            "Use lorafusion.ops.mock_ops for CPU/testing, or run "
            "colab_verification/run_triton_tests.ipynb on a GPU runtime."
        )


# --------------------------------------------------------------------------
# Host-side operand preparation (fp16 + orientation), memoized for weights.
# --------------------------------------------------------------------------

_PREP_CACHE: dict[tuple[int, bool], tuple] = {}
_PREP_CACHE_MAX = 256


def clear_weight_cache() -> None:
    """Drop all memoized fp16/transposed weight copies.

    `_prepare` invalidates itself correctly on ordinary mutation (see below),
    so this is only needed to reclaim memory.
    """
    _PREP_CACHE.clear()


def _prepare(t: torch.Tensor, transpose: bool) -> torch.Tensor:
    """Return `t` (or `t.T`) as a contiguous fp16 tensor, memoized.

    Intended for the static tensors -- `base_weight`, `A`, `B` -- where the
    transpose+cast would otherwise be repeated on every call.

    The cache entry is keyed on `id(t)` and validated against three things
    before being trusted, because `id()` alone is unsafe:
      * a weakref identity check, so a recycled `id()` from a freed tensor
        cannot alias onto a stale entry;
      * `data_ptr()`, which catches `param.data = <new tensor>` rebinds that
        do not bump the version counter;
      * `_version`, which catches in-place mutation (`optimizer.step()`,
        `copy_`, `add_`) of the same storage.
    """
    key = (id(t), transpose)
    entry = _PREP_CACHE.get(key)
    if entry is not None:
        ref, ptr, version, cached = entry
        if ref() is t and ptr == t.data_ptr() and version == t._version:
            return cached

    view = t.t() if transpose else t
    prepared = view.to(torch.float16).contiguous()

    if len(_PREP_CACHE) >= _PREP_CACHE_MAX:
        _PREP_CACHE.clear()
    try:
        _PREP_CACHE[key] = (weakref.ref(t), t.data_ptr(), t._version, prepared)
    except TypeError:  # pragma: no cover - non-weakref-able tensor subclass
        pass
    return prepared


def _as_fp16(t: torch.Tensor) -> torch.Tensor:
    """Cheap fp16 view/copy for *activations* (never cached -- they change)."""
    if t.dtype == torch.float16 and t.is_contiguous():
        return t
    return t.to(torch.float16).contiguous()


if TRITON_AVAILABLE:

    # Sized for fp16 operands with the LoRA branch hoisted out of the K-loop,
    # so a stage costs (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N)*2 bytes. num_stages
    # is chosen per-tile to stay under a Turing SM's 64 KB: a big tile with a
    # deep pipeline silently exceeds it and gets PRUNED by the autotuner, which
    # is how this kernel previously ended up running on 64x64 tiles. Every
    # config below is <=56 KB on T4.
    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
        # Fallback for the tiny shapes used by the correctness tests.
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_SIZE_M": 4}, num_warps=2, num_stages=2),
    ]

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "K1", "N", "R"])
    @triton.jit
    def _fused_double_gemm_kernel(
        p_ptr,
        m1_ptr,
        s_ptr,
        m3_ptr,
        out_ptr,
        scale,
        M,
        K1,
        N,
        R,
        stride_pm,
        stride_pk,
        stride_m1k,
        stride_m1n,
        stride_sm,
        stride_sr,
        stride_m3r,
        stride_m3n,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ) -> None:
        """out[m,n] = sum_k P[m,k]*M1[k,n] + scale * sum_r S[m,r]*M3[r,n]

        P (M,K1), M1 (K1,N), S (M,R), M3 (R,N) all arrive fp16, contiguous,
        already in this exact logical orientation -- the caller does any
        transposing (see `_prepare`). Every load is coalesced; there is no
        in-kernel transpose.

        The K-loop contains exactly ONE `tl.dot`. That is deliberate and is
        the difference between ~1.5 and ~15 TFLOPS here. The obvious way to
        write this op fuses the LoRA down-projection into the same loop:

            acc   = tl.dot(p_tile, m1_tile, acc)      # (BM,BK)x(BK,BN)
            s_acc = tl.dot(p_tile, m2_tile, s_acc)    # (BM,BK)x(BK,16)

        but those two dots want `p_tile` in two *different* MMA operand
        layouts (the N dimensions differ, 128 vs 16). Triton cannot hold one
        register tile in both, so it inserts a layout conversion -- a full
        round-trip of `p_tile` through shared memory -- on every single K
        iteration. It also recomputes the entire down-projection redundantly
        in every one of the N/BLOCK_N column programs.

        So the down-projection S = P @ M2 is hoisted out and computed once,
        up front (it is a rank-R skinny GEMM, ~0.4% of the FLOPs here). The
        loop below is then a clean, textbook, single-layout matmul, and the
        LoRA branch costs one rank-R `tl.dot` in the epilogue.

        The fusion that matters is still intact: `out` is written exactly
        once, with the delta already folded in. An unfused implementation
        materialises the (M,N) delta and does a second full read-modify-write
        pass over the output.

        One program instance computes one (BLOCK_M, BLOCK_N) output tile for
        a single adapter route. Assumes R <= BLOCK_R (LoRA rank is small).

        Uses a 1D "grouped" launch grid (Triton matmul-tutorial trick): a
        naive row-major pid->tile mapping visits tiles in an order that
        reloads M1 column-tiles from L2 constantly. Grouping several
        row-blocks together and sweeping columns within the group instead
        keeps recently-used column tiles in L2 across neighboring programs.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Offsets stay plain `pid*BLOCK + arange` and out-of-range lanes are
        # handled with masks. Do NOT "optimise" this into `% M` / `% N`: the
        # modulo is applied to `offs_n`, which indexes the *fast-varying*,
        # stride-1 dimension of the M1 load. Triton vectorises a load into a
        # wide 128-bit transaction only when it can prove the lane offsets
        # are contiguous, and `%` destroys that proof -- every coalesced
        # weight load silently degrades into a per-lane gather. Measured on a
        # T4 that cost ~8x: 2.1 TFLOP/s, which is neither compute-bound (65
        # peak) nor bandwidth-bound (~25 GB/s of 320), the signature of
        # uncoalesced access. Masked loads keep full coalescing, because
        # Triton emits a predicated vector load rather than a gather.
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, BLOCK_K)

        # Loop-invariant halves of the load masks, hoisted out of the K-loop.
        mask_m = offs_m[:, None] < M
        mask_n = offs_n[None, :] < N

        p_ptrs = p_ptr + offs_m[:, None] * stride_pm + offs_k[None, :] * stride_pk
        m1_ptrs = m1_ptr + offs_k[:, None] * stride_m1k + offs_n[None, :] * stride_m1n

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K1, BLOCK_K)):
            k_remaining = K1 - k * BLOCK_K
            p_tile = tl.load(
                p_ptrs, mask=mask_m & (offs_k[None, :] < k_remaining), other=0.0
            )
            m1_tile = tl.load(
                m1_ptrs, mask=mask_n & (offs_k[:, None] < k_remaining), other=0.0
            )
            # fp16 tensor cores, fp32 accumulate, matching the paper's
            # mixed-precision setup. (Not TF32: it needs Ampere+ and silently
            # no-ops on Turing.)
            acc = tl.dot(p_tile, m1_tile, acc)
            p_ptrs += BLOCK_K * stride_pk
            m1_ptrs += BLOCK_K * stride_m1k

        # Rank-R LoRA epilogue, folded in before the single store.
        s_tile = tl.load(
            s_ptr + offs_m[:, None] * stride_sm + offs_r[None, :] * stride_sr,
            mask=mask_m & (offs_r[None, :] < R),
            other=0.0,
        )
        m3_tile = tl.load(
            m3_ptr + offs_r[:, None] * stride_m3r + offs_n[None, :] * stride_m3n,
            mask=mask_n & (offs_r[:, None] < R),
            other=0.0,
        )
        acc += scale * tl.dot(s_tile, m3_tile)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m & mask_n,
        )


def _launch_fused_double_gemm(
    p: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    m3: torch.Tensor,
    scale: float,
    out: torch.Tensor,
) -> None:
    """out[:] = p @ m1 + scale * (p @ m2) @ m3.

    All inputs must already be fp16 and in the kernel's logical orientation:
    p (M,K1), m1 (K1,N), m2 (K1,R), m3 (R,N). `out` is written in place, so
    callers can pass a row-slice view of a larger tensor and skip a
    temporary allocation plus a full (M,N) copy.

    The rank-R down-projection `p @ m2` is done here, once, by cuBLAS rather
    than inside the kernel's K-loop -- see the kernel docstring for why. It
    is a skinny (M,K1)x(K1,R) GEMM: at rank 16 with K1=N=4096 it is 0.4% of
    the op's FLOPs, and cuBLAS accumulates it in fp32 on tensor cores.
    """
    M, K1 = p.shape
    K1a, N = m1.shape
    K1b, R = m2.shape
    Rb, Nb = m3.shape
    assert K1 == K1a == K1b and N == Nb and R == Rb

    s = p @ m2  # (M, R), fp16

    BLOCK_R = triton.next_power_of_2(max(R, 16))

    # 1D grid: required for the grouped pid_m/pid_n swizzle inside the
    # kernel, which trades a naive row-major tile order for one that keeps
    # column tiles hot in L2 across neighboring programs.
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _fused_double_gemm_kernel[grid](
        p,
        m1,
        s,
        m3,
        out,
        scale,
        M,
        K1,
        N,
        R,
        p.stride(0),
        p.stride(1),
        m1.stride(0),
        m1.stride(1),
        s.stride(0),
        s.stride(1),
        m3.stride(0),
        m3.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_R=BLOCK_R,
    )


def fused_lora_forward(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_a: dict[int, torch.Tensor],
    lora_b: dict[int, torch.Tensor],
    adapters: dict[int, AdapterSpec],
    routing: TileRoutingConfig,
) -> torch.Tensor:
    """GPU fused forward. Same signature/semantics as `mock_ops.fused_lora_forward`."""
    _require_triton()
    routing.validate(x.shape[0])

    out_features = base_weight.shape[0]
    # Output dtype follows the input. The paper's whole mechanism is saving
    # DRAM traffic on full-sized activation tensors, so forcing an fp32 (M,N)
    # output would hand back half of what the fusion just saved. fp32 in ->
    # fp32 out keeps the pure-fp32 mock reference comparison exact.
    out = torch.empty((x.shape[0], out_features), device=x.device, dtype=x.dtype)

    # Cast the whole activation batch once, then slice -- a row slice of a
    # contiguous 2D tensor is itself contiguous, so no per-route copy.
    x16 = _as_fp16(x)
    # W^T: (in, out). Memoized -- base_weight is frozen.
    w_t = _prepare(base_weight, transpose=True)

    for route in routing.routes:
        adapter = adapters[route.adapter_id]
        _launch_fused_double_gemm(
            x16[route.row_start : route.row_end],
            w_t,
            _prepare(lora_a[route.adapter_id], transpose=True),  # A^T: (in, rank)
            _prepare(lora_b[route.adapter_id], transpose=True),  # B^T: (rank, out)
            adapter.scaling,
            out[route.row_start : route.row_end],
        )

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
    """GPU fused backward. Same signature/semantics as `mock_ops.fused_lora_backward`."""
    _require_triton()
    routing.validate(x.shape[0])

    grad_x = torch.empty_like(x)
    grad_lora_a: dict[int, torch.Tensor] = {}
    grad_lora_b: dict[int, torch.Tensor] = {}

    g16 = _as_fp16(grad_out)
    # dX = dY @ W + scale * (dY @ B) @ A. W:(out,in), B:(out,rank),
    # A:(rank,in) are already in the exact orientation the algebra needs, so
    # `_prepare` here is a pure fp16 cast with no transpose.
    w = _prepare(base_weight, transpose=False)

    for route in routing.routes:
        adapter = adapters[route.adapter_id]
        scaling = adapter.scaling
        x_slice = x[route.row_start : route.row_end]
        g_slice = grad_out[route.row_start : route.row_end]
        a = lora_a[route.adapter_id]
        b = lora_b[route.adapter_id]

        _launch_fused_double_gemm(
            g16[route.row_start : route.row_end],
            w,
            _prepare(b, transpose=False),
            _prepare(a, transpose=False),
            scaling,
            grad_x[route.row_start : route.row_end],
        )

        # dA, dB: small-tensor path, left unfused (matches paper design).
        # Deliberately kept in the input dtype rather than fp16: `grad_a`
        # reduces over the full row dimension, which can be thousands of
        # rows, and that is exactly where fp16 accumulation loses precision.
        xa = x_slice @ a.t()  # (rows, rank)
        grad_b = scaling * (g_slice.t() @ xa)
        grad_xa = scaling * (g_slice @ b)
        grad_a = grad_xa.t() @ x_slice

        grad_lora_a[route.adapter_id] = (
            grad_lora_a.get(route.adapter_id, 0) + grad_a
        )
        grad_lora_b[route.adapter_id] = (
            grad_lora_b.get(route.adapter_id, 0) + grad_b
        )

    return grad_x, grad_lora_a, grad_lora_b
