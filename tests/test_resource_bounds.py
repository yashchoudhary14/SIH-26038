"""Peak memory must not scale with cohort size.

The failure this guards is not a crash in the code under test: it is a
`_ArrayMemoryError` on a 1 MiB allocation thousands of cases later, inside
OpenCV, on a machine that has plenty of memory for any single case.
"""
from __future__ import annotations

import importlib.util
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "precompute_features",
    Path(__file__).resolve().parents[1] / "scripts" / "precompute_features.py")
precompute_features = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(precompute_features)
bounded_map = precompute_features.bounded_map


def test_bounded_map_keeps_results_in_order():
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(bounded_map(pool, lambda x: x * 2, range(200), 8))
    assert got == [x * 2 for x in range(200)]


def test_bounded_map_does_not_buffer_the_whole_input():
    """The whole point: unconsumed results must not pile up.

    ``ThreadPoolExecutor.map`` submits every task immediately, so with a slow
    consumer the finished results accumulate. Each one here carries a ~12 MB
    tensor in the real pipeline, which is how a 12,495-case cohort exhausted
    32 GB of RAM. Track how many inputs the pool has pulled before the consumer
    has taken anything.
    """
    started = []
    lock = threading.Lock()

    def work(x):
        with lock:
            started.append(x)
        return x

    with ThreadPoolExecutor(max_workers=2) as pool:
        gen = bounded_map(pool, work, range(1000), 8)
        first = next(gen)          # consume exactly one result
        time.sleep(0.05)           # give the pool a chance to run ahead
        with lock:
            ran_ahead = len(started)
        assert first == 0
        assert ran_ahead <= 16, (
            f"pool ran {ran_ahead} tasks ahead of a consumer that took 1; "
            "the in-flight window is not bounded, so peak memory scales with "
            "the cohort")
        list(gen)                  # drain so the pool shuts down cleanly


def test_bounded_map_handles_an_input_shorter_than_the_window():
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(bounded_map(pool, lambda x: x, range(3), 64)) == [0, 1, 2]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(bounded_map(pool, lambda x: x, [], 64)) == []


def test_bounded_map_propagates_worker_exceptions():
    def boom(x):
        if x == 5:
            raise ValueError("worker failed")
        return x

    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(ValueError, match="worker failed"):
            list(bounded_map(pool, boom, range(50), 8))
