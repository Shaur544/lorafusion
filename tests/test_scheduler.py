"""Bin-packer optimality and bubble-lemma safety tests."""

from __future__ import annotations

from lorafusion.scheduler.adapter_grouper import (
    Request,
    bucket_by_adapter,
    check_bubble_lemma,
    group_by_adapter,
)
from lorafusion.scheduler.milp_packer import greedy_pack, milp_pack


def test_greedy_pack_respects_capacity():
    adapter_rows = {0: 100, 1: 50, 2: 30}
    result = greedy_pack(adapter_rows, tile_capacity=64)

    for tile in result.bins:
        assert sum(rows for _, rows in tile) <= 64

    total_packed = sum(rows for tile in result.bins for _, rows in tile)
    assert total_packed == sum(adapter_rows.values())


def test_greedy_pack_splits_oversized_adapter():
    result = greedy_pack({0: 150}, tile_capacity=64)
    rows_per_bin = [sum(r for _, r in tile) for tile in result.bins]
    assert all(r <= 64 for r in rows_per_bin)
    assert sum(rows_per_bin) == 150


def test_milp_pack_no_worse_than_greedy_bins():
    adapter_rows = {0: 40, 1: 40, 2: 40, 3: 40}
    greedy_result = greedy_pack(adapter_rows, tile_capacity=64)
    milp_result = milp_pack(adapter_rows, tile_capacity=64, time_limit_s=5.0)

    assert milp_result.num_bins <= greedy_result.num_bins


def test_milp_pack_falls_back_gracefully():
    # Should not raise even if PuLP is unavailable or times out; falls back
    # to a valid (if suboptimal) packing.
    result = milp_pack({0: 10, 1: 200}, tile_capacity=64, time_limit_s=1.0)
    total_packed = sum(rows for tile in result.bins for _, rows in tile)
    assert total_packed == 210
    for tile in result.bins:
        assert sum(rows for _, rows in tile) <= 64


def test_group_by_adapter_preserves_contiguous_runs():
    requests = [
        Request(0, adapter_id=1, num_rows=4),
        Request(1, adapter_id=1, num_rows=6),
        Request(2, adapter_id=2, num_rows=3),
        Request(3, adapter_id=1, num_rows=2),
    ]
    groups = group_by_adapter(requests)

    assert len(groups) == 3
    assert groups[0].adapter_id == 1 and groups[0].total_rows == 10
    assert groups[1].adapter_id == 2 and groups[1].total_rows == 3
    assert groups[2].adapter_id == 1 and groups[2].total_rows == 2


def test_bucket_by_adapter_merges_all_occurrences():
    requests = [
        Request(0, adapter_id=1, num_rows=4),
        Request(1, adapter_id=2, num_rows=6),
        Request(2, adapter_id=1, num_rows=3),
    ]
    groups = bucket_by_adapter(requests)

    assert len(groups) == 2
    by_adapter = {g.adapter_id: g.total_rows for g in groups}
    assert by_adapter[1] == 7
    assert by_adapter[2] == 6


def test_bubble_lemma_holds_for_valid_grouping():
    requests = [
        Request(0, adapter_id=1, num_rows=4),
        Request(1, adapter_id=1, num_rows=6),
        Request(2, adapter_id=2, num_rows=3),
    ]
    groups = group_by_adapter(requests)
    assert check_bubble_lemma(groups, requests)


def test_bubble_lemma_detects_split_request():
    requests = [Request(0, adapter_id=1, num_rows=10)]
    # Simulate an unsafe grouping that double-counts/splits request 0's rows.
    from lorafusion.scheduler.adapter_grouper import AdapterGroup

    bad_groups = [
        AdapterGroup(adapter_id=1, request_ids=[0], total_rows=6),
        AdapterGroup(adapter_id=1, request_ids=[0], total_rows=4),
    ]
    assert not check_bubble_lemma(bad_groups, requests)
