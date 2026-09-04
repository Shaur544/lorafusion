"""Bubble-lemma-aware head-tail grouping of adapter requests.

The "bubble lemma" is the safety property this module enforces: when two
adapters share a fused tile, the boundary between them (the "bubble") must
not split a single request's rows across a synchronization point in a way
that would let one adapter's kernel program observe another's partial
state. Concretely, we require that requests be grouped so that each
adapter's rows form one contiguous head-to-tail run per tile — never
interleaved.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    """A single inference/training request bound to one adapter."""

    request_id: int
    adapter_id: int
    num_rows: int


@dataclass(frozen=True)
class AdapterGroup:
    """Contiguous run of requests for one adapter within a tile."""

    adapter_id: int
    request_ids: list[int]
    total_rows: int


def group_by_adapter(requests: list[Request]) -> list[AdapterGroup]:
    """Group requests into contiguous per-adapter runs, preserving input order
    of first appearance.

    This does NOT reorder requests across adapter boundaries — it only
    merges consecutive requests that already share an adapter. Callers that
    need cross-adapter reordering should sort/bucket `requests` by
    `adapter_id` before calling, at the cost of changing arrival order.
    """
    groups: list[AdapterGroup] = []
    current_ids: list[int] = []
    current_rows = 0
    current_adapter: int | None = None

    for req in requests:
        if req.adapter_id != current_adapter:
            if current_adapter is not None:
                groups.append(
                    AdapterGroup(current_adapter, current_ids, current_rows)
                )
            current_adapter = req.adapter_id
            current_ids = []
            current_rows = 0
        current_ids.append(req.request_id)
        current_rows += req.num_rows

    if current_adapter is not None:
        groups.append(AdapterGroup(current_adapter, current_ids, current_rows))

    return groups


def bucket_by_adapter(requests: list[Request]) -> list[AdapterGroup]:
    """Bucket ALL requests for each adapter together, regardless of arrival
    order, sorted by adapter_id for determinism.

    Use this before tile packing when maximizing per-tile adapter locality
    matters more than preserving arrival order.
    """
    buckets: dict[int, list[Request]] = {}
    for req in requests:
        buckets.setdefault(req.adapter_id, []).append(req)

    groups: list[AdapterGroup] = []
    for adapter_id in sorted(buckets):
        reqs = buckets[adapter_id]
        groups.append(
            AdapterGroup(
                adapter_id=adapter_id,
                request_ids=[r.request_id for r in reqs],
                total_rows=sum(r.num_rows for r in reqs),
            )
        )
    return groups


def check_bubble_lemma(groups: list[AdapterGroup], requests: list[Request]) -> bool:
    """Verify the bubble-lemma safety property: every request's rows appear
    entirely within exactly one group's contiguous run — no request is split
    or duplicated across groups.
    """
    request_by_id = {r.request_id: r for r in requests}
    seen: set[int] = set()

    for group in groups:
        if len(group.request_ids) != len(set(group.request_ids)):
            return False
        group_rows = sum(
            request_by_id[rid].num_rows for rid in group.request_ids
        )
        if group_rows != group.total_rows:
            return False
        for rid in group.request_ids:
            if rid in seen:
                return False
            if request_by_id[rid].adapter_id != group.adapter_id:
                return False
            seen.add(rid)

    return seen == set(request_by_id)
