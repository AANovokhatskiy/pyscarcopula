"""Opt-in acceptance tests for native parallel release gates."""

from __future__ import annotations

import json
import ctypes
import multiprocessing
import os
import subprocess
import sys

import pytest


def _record_metric(name, payload):
    output = os.environ.get("PYSCA_RELEASE_METRICS_OUTPUT")
    if not output:
        return
    path = os.path.abspath(output)
    try:
        with open(path, encoding="utf-8") as stream:
            report = json.load(stream)
    except FileNotFoundError:
        report = {"status": "passed", "metrics": {}}
    report["metrics"][name] = payload
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _allocation_probe():
    path = os.environ.get("PYSCA_ALLOCATION_PROBE")
    if not path:
        pytest.skip("PYSCA_ALLOCATION_PROBE is not configured")
    probe = ctypes.CDLL(path)
    probe.pysca_allocation_probe_enable.argtypes = [ctypes.c_int]
    for name in (
        "pysca_allocation_probe_calls",
        "pysca_allocation_probe_bytes",
        "pysca_allocation_probe_frees",
    ):
        getattr(probe, name).restype = ctypes.c_ulonglong
    return probe


def _measure_allocations(probe, function):
    probe.pysca_allocation_probe_enable(0)
    probe.pysca_allocation_probe_reset()
    probe.pysca_allocation_probe_enable(1)
    try:
        result = function()
    finally:
        probe.pysca_allocation_probe_enable(0)
    snapshot = {
        "calls": probe.pysca_allocation_probe_calls(),
        "bytes": probe.pysca_allocation_probe_bytes(),
        "frees": probe.pysca_allocation_probe_frees(),
    }
    return result, snapshot


def _lifecycle_child(connection, n_threads, exercise_failure):
    from pyscarcopula.numerical import _cpp_extension

    module = _cpp_extension.load()
    try:
        if exercise_failure and n_threads > 1:
            try:
                module._parallel_for_blocks_probe(32, 1, n_threads, 1)
            except RuntimeError:
                pass
            else:
                raise AssertionError("worker failure was not propagated")
        first = dict(module._parallel_for_blocks_probe(
            32, 1, n_threads))
        sequential = dict(module._parallel_for_blocks_probe(16, 1, 1))
        second = dict(module._parallel_for_blocks_probe(
            32, 1, n_threads))
        connection.send({
            "pid": os.getpid(),
            "first": first,
            "sequential": sequential,
            "second": second,
        })
    finally:
        connection.close()


def _run_lifecycle_child(start_method, n_threads, exercise_failure=False):
    if start_method == "fork":
        # Deliberately fork after the parent has live native workers. The child
        # must ignore the inherited runtime and publish its own pool only when
        # n_threads > 1 is requested.
        from pyscarcopula.numerical import _cpp_extension

        _cpp_extension.load()._parallel_for_blocks_probe(32, 1, 4)
    context = multiprocessing.get_context(start_method)
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_lifecycle_child,
        args=(child, n_threads, exercise_failure),
    )
    process.start()
    child.close()
    assert parent.poll(30), (
        f"{start_method} child timed out with n_threads={n_threads}")
    result = parent.recv()
    parent.close()
    process.join(timeout=30)
    assert process.exitcode == 0
    return result


@pytest.mark.validation
@pytest.mark.skipif(os.name == "nt", reason="Unix lifecycle gate")
@pytest.mark.parametrize("start_method", ["spawn", "fork", "forkserver"])
@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_unix_start_methods_own_parallel_runtime(start_method, n_threads):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is unavailable")

    if n_threads > 1:
        from pyscarcopula.numerical import _cpp_extension

        module = _cpp_extension.load()
        with pytest.raises(RuntimeError, match="requested failure"):
            module._parallel_for_blocks_probe(32, 1, n_threads, 1)
        recovered = dict(module._parallel_for_blocks_probe(
            32, 1, n_threads))
        assert recovered["runtime"]["initialized"] is True

    result = _run_lifecycle_child(
        start_method, n_threads, exercise_failure=True)
    first_runtime = result["first"]["runtime"]
    second_runtime = result["second"]["runtime"]
    if n_threads == 1:
        assert first_runtime["initialized"] is False
        assert second_runtime["initialized"] is False
    else:
        assert first_runtime["owner_pid"] == result["pid"]
        assert second_runtime["owner_pid"] == result["pid"]
        assert first_runtime["worker_count"] == n_threads
        assert second_runtime["worker_count"] == n_threads
        assert (
            second_runtime["batches_submitted"]
            > first_runtime["batches_submitted"]
        )


@pytest.mark.validation
@pytest.mark.skipif(os.name == "nt", reason="Unix lifecycle gate")
def test_process_lifecycle_stress_has_no_timeouts_or_orphans():
    cycles = int(os.environ.get("PYSCA_RELEASE_STRESS_CYCLES", "100"))
    assert cycles >= 100
    start_method = (
        "forkserver"
        if "forkserver" in multiprocessing.get_all_start_methods()
        else "spawn"
    )
    last_result = None
    for cycle in range(cycles):
        n_threads = (1, 2, 4)[cycle % 3]
        last_result = _run_lifecycle_child(
            start_method,
            n_threads,
            exercise_failure=(cycle % 10 == 0),
        )
        assert last_result["pid"] > 0
    assert not multiprocessing.active_children()
    _record_metric("process_lifecycle", {
        "cycles": cycles,
        "start_method": start_method,
        "remaining_children": 0,
        "last_runtime": last_result["second"]["runtime"],
    })


@pytest.mark.validation
def test_subinterpreter_contract_is_immediate_rejection():
    source = (
        "import json\n"
        "try:\n"
        "    import _xxsubinterpreters as interpreters\n"
        "except ImportError:\n"
        "    print(json.dumps({'status': 'unavailable'}))\n"
        "    raise SystemExit(0)\n"
        "import pyscarcopula._scar_cpp\n"
        "interpreter = interpreters.create()\n"
        "try:\n"
        "    try:\n"
        "        interpreters.run_string("
        "interpreter, 'import pyscarcopula._scar_cpp')\n"
        "    except interpreters.RunFailedError as exc:\n"
        "        print(json.dumps({'status': 'rejected', "
        "'message': str(exc)}))\n"
        "    else:\n"
        "        print(json.dumps({'status': 'accepted'}))\n"
        "finally:\n"
        "    interpreters.destroy(interpreter)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(completed.stdout)
    if payload["status"] == "unavailable":
        pytest.skip("_xxsubinterpreters is unavailable")
    assert payload["status"] == "rejected"
    assert "interpreter" in payload["message"].lower()
    _record_metric("subinterpreters", payload)


@pytest.mark.validation
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="/proc RSS gate requires Linux",
)
def test_repeated_native_hot_paths_have_bounded_resident_memory():
    source = (
        "import gc, json, os\n"
        "import numpy as np\n"
        "from pyscarcopula import EquicorrGaussianCopula, "
        "StochasticStudentCopula\n"
        "from pyscarcopula.numerical import static_likelihood\n"
        "def rss():\n"
        "    with open('/proc/self/statm', encoding='ascii') as stream:\n"
        "        pages = int(stream.read().split()[1])\n"
        "    return pages * os.sysconf('SC_PAGE_SIZE')\n"
        "rng = np.random.default_rng(913)\n"
        "u = rng.uniform(0.05, 0.95, (32, 8))\n"
        "grid = np.linspace(-1.0, 1.0, 12)\n"
        "R = np.full((8, 8), 0.1)\n"
        "np.fill_diagonal(R, 1.0)\n"
        "student = StochasticStudentCopula(d=8, R=R)\n"
        "equicorr = EquicorrGaussianCopula(d=8)\n"
        "prepared = static_likelihood.prepare("
        "student, u, n_threads=4)\n"
        "def exercise():\n"
        "    student.pdf_and_grad_on_grid_batch("
        "u, grid, n_threads=4)\n"
        "    equicorr.pdf_and_grad_on_grid_batch("
        "u, grid, n_threads=4)\n"
        "    prepared.joint_result(6.0)\n"
        "for _ in range(20): exercise()\n"
        "gc.collect()\n"
        "before = rss()\n"
        "for _ in range(1000): exercise()\n"
        "gc.collect()\n"
        "after = rss()\n"
        "print(json.dumps({'before': before, 'after': after, "
        "'growth': after - before}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(completed.stdout)
    assert payload["growth"] <= 32 * 1024 * 1024
    _record_metric("resident_memory", {
        **payload,
        "limit": 32 * 1024 * 1024,
        "repeated_calls": 1000,
    })


@pytest.mark.validation
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="glibc allocation probe requires Linux",
)
def test_native_grid_has_no_allocation_per_grid_cell():
    import numpy as np

    from pyscarcopula import EquicorrGaussianCopula
    from pyscarcopula.numerical import _cpp_copula, _cpp_extension

    probe = _allocation_probe()
    module = _cpp_extension.load()
    copula = EquicorrGaussianCopula(d=20)
    observations = np.random.default_rng(914).uniform(
        0.05, 0.95, (4160, 20))
    spec = _cpp_copula.make_multivariate_spec(module, copula, cache=None)

    def evaluate(grid):
        return module.multivariate_pdf_and_grad_grid(
            spec, observations, grid, 0, 4)

    evaluate(np.linspace(-1.0, 1.0, 8))
    _, small = _measure_allocations(
        probe, lambda: evaluate(np.linspace(-1.0, 1.0, 8)))
    _, large = _measure_allocations(
        probe, lambda: evaluate(np.linspace(-1.0, 1.0, 64)))

    # Output bytes scale with T*K, but the number of allocations must remain
    # bounded rather than growing per grid cell.
    assert large["calls"] <= small["calls"] + 32
    assert large["calls"] < 256
    assert large["bytes"] > small["bytes"]
    _record_metric("allocation_probe", {
        "small_grid": small,
        "large_grid": large,
        "max_call_growth": 32,
        "max_large_calls": 255,
    })
