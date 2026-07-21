"""Phase-1 contracts for the native per-process thread runtime."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import multiprocessing
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from pyscarcopula import (
    EquicorrGaussianCopula,
    GaussianCopula,
    NumericalConfig,
    StochasticStudentCopula,
    StudentCopula,
)
from pyscarcopula.numerical import _cpp_extension, _cpp_scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


def _runtime_child_probe(queue, n_threads):
    module = _cpp_extension.load()
    module._parallel_for_blocks_probe(16, 1, n_threads)
    queue.put(dict(module._parallel_runtime_info()))


def _run_clean_interpreter(source):
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


def test_n_threads_one_never_initializes_runtime():
    payload = _run_clean_interpreter(
        "import json\n"
        "from pyscarcopula.numerical import _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "before = dict(m._parallel_runtime_info())\n"
        "result = dict(m._parallel_for_blocks_probe(100, 1, 1))\n"
        "after = dict(m._parallel_runtime_info())\n"
        "print(json.dumps({'before': before, 'result': result, "
        "'after': after}))\n"
    )
    assert payload["before"]["initialized"] is False
    assert payload["after"]["initialized"] is False
    assert payload["result"]["block_ids"] == [0] * 100


def test_spawned_interpreter_n_threads_one_ignores_parent_pool():
    payload = _run_clean_interpreter(
        "import json, subprocess, sys\n"
        "from pyscarcopula.numerical import _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "m._parallel_for_blocks_probe(32, 1, 4)\n"
        "source = ('import json\\n' "
        "+ 'from pyscarcopula.numerical import _cpp_extension\\n' "
        "+ 'm = _cpp_extension.load()\\n' "
        "+ 'm._parallel_for_blocks_probe(16, 1, 1)\\n' "
        "+ 'print(json.dumps(dict(m._parallel_runtime_info())))\\n')\n"
        "child = subprocess.run([sys.executable, '-c', source], "
        "check=True, capture_output=True, text=True, timeout=20)\n"
        "print(child.stdout, end='')\n"
    )
    assert payload["initialized"] is False


def test_runtime_uses_stable_blocks_and_reuses_workers():
    module = _cpp_extension.load()
    before = dict(module._parallel_runtime_info())
    first = dict(module._parallel_for_blocks_probe(10, 2, 3))
    second = dict(module._parallel_for_blocks_probe(10, 2, 2))

    assert first["block_ids"] == [0] * 4 + [1] * 3 + [2] * 3
    assert second["block_ids"] == [0] * 5 + [1] * 5
    assert first["runtime"]["worker_count"] >= 3
    assert second["runtime"]["worker_count"] == first["runtime"]["worker_count"]
    assert second["runtime"]["batches_submitted"] == (
        first["runtime"]["batches_submitted"] + 1)
    assert first["runtime"]["batches_submitted"] >= (
        before["batches_submitted"] + 1)

    batches = second["runtime"]["batches_submitted"]
    module._parallel_for_blocks_probe(100, 1, 1)
    assert dict(module._parallel_runtime_info())["batches_submitted"] == batches


def test_worker_exception_is_rethrown_and_runtime_recovers():
    module = _cpp_extension.load()
    with pytest.raises(RuntimeError, match="parallel probe requested failure"):
        module._parallel_for_blocks_probe(12, 1, 4, 2)

    recovered = dict(module._parallel_for_blocks_probe(12, 1, 4))
    assert recovered["block_ids"] == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3


def test_nested_parallel_call_uses_worker_local_sequential_path():
    module = _cpp_extension.load()
    before = dict(module._parallel_runtime_info())["batches_submitted"]
    result = dict(module._parallel_for_blocks_probe(
        16, 1, 4, nested_threads=4))

    assert result["block_ids"] == (
        [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    assert result["runtime"]["batches_submitted"] == before + 1


def test_runtime_shutdown_is_idempotent_and_pool_can_be_recreated():
    module = _cpp_extension.load()
    module._parallel_for_blocks_probe(16, 1, 3)

    assert dict(module._parallel_runtime_shutdown())["initialized"] is False
    assert dict(module._parallel_runtime_shutdown())["initialized"] is False

    recreated = dict(module._parallel_for_blocks_probe(16, 1, 2))
    assert recreated["runtime"]["initialized"] is True
    assert recreated["runtime"]["worker_count"] == 2
    assert recreated["runtime"]["batches_submitted"] == 1


def test_repeated_interpreter_process_teardown_after_parallel_work():
    source = (
        "import json\n"
        "from pyscarcopula.numerical import _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "m._parallel_for_blocks_probe(32, 1, 4)\n"
        "print(json.dumps(dict(m._parallel_runtime_info())))\n"
    )
    for _ in range(3):
        payload = _run_clean_interpreter(source)
        assert payload["initialized"] is True
        assert payload["worker_count"] == 4


def test_runtime_accepts_concurrent_independent_callers():
    module = _cpp_extension.load()

    def call(size):
        return dict(module._parallel_for_blocks_probe(size, 2, 4))["block_ids"]

    sizes = [17, 23, 31, 41]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(call, sizes))

    for size, block_ids in zip(sizes, results):
        assert len(block_ids) == size
        assert block_ids == sorted(block_ids)
        assert set(block_ids) == {0, 1, 2, 3}


def test_numerical_config_default_ignores_environment(monkeypatch):
    monkeypatch.setenv("PYSCARCOPULA_NUM_THREADS", "6")
    assert NumericalConfig().n_threads == 1
    assert NumericalConfig(n_threads=3).n_threads == 3

    native = _cpp_scar_ou._config(
        _cpp_extension.load(), AutoTMConfig(n_threads=5))
    assert native.n_threads == 5

    monkeypatch.setenv("PYSCARCOPULA_NUM_THREADS", "invalid")
    assert NumericalConfig().n_threads == 1


def test_import_time_environment_cannot_enable_parallelism():
    payload = _run_clean_interpreter(
        "import json, os\n"
        "os.environ['PYSCARCOPULA_NUM_THREADS'] = '8'\n"
        "from pyscarcopula import NumericalConfig\n"
        "from pyscarcopula._types import DEFAULT_CONFIG\n"
        "from pyscarcopula.numerical import _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "print(json.dumps({\n"
        "    'config': NumericalConfig().n_threads,\n"
        "    'default': DEFAULT_CONFIG.n_threads,\n"
        "    'runtime': dict(m._parallel_runtime_info()),\n"
        "}))\n"
    )
    assert payload["config"] == 1
    assert payload["default"] == 1
    assert payload["runtime"]["initialized"] is False


def test_multivariate_thread_arguments_have_absolute_one_default():
    classes = (
        EquicorrGaussianCopula,
        GaussianCopula,
        StochasticStudentCopula,
        StudentCopula,
    )
    checked = []
    for cls in classes:
        for name, member in inspect.getmembers(cls, callable):
            try:
                parameter = inspect.signature(member).parameters.get(
                    "n_threads")
            except (TypeError, ValueError):
                continue
            if parameter is not None:
                checked.append(f"{cls.__name__}.{name}")
                assert parameter.default == 1

    assert checked


@pytest.mark.parametrize(
    "value", [None, 0, -1, 257, True, 1.5, "invalid"])
def test_numerical_config_rejects_invalid_thread_count(value):
    with pytest.raises(ValueError, match="n_threads"):
        NumericalConfig(n_threads=value)


def test_fit_diagnostics_record_resolved_thread_count():
    rng = np.random.default_rng(601)
    u = rng.uniform(0.05, 0.95, (40, 3))
    result = EquicorrGaussianCopula(d=3).fit(
        u,
        method="mle",
        config=NumericalConfig(n_threads=3),
        maxiter=5,
    )
    assert result.diagnostics["n_threads"] == 3


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("gas", {"maxiter": 1, "maxfun": 5}),
        (
            "scar-tm-ou",
            {
                "K": 8,
                "max_K": 8,
                "maxiter": 1,
                "maxfun": 5,
                "transition_method": "matrix",
            },
        ),
    ],
)
def test_dynamic_fit_diagnostics_record_resolved_thread_count(method, kwargs):
    u = np.random.default_rng(606).uniform(0.05, 0.95, (20, 3))
    result = EquicorrGaussianCopula(d=3).fit(
        u,
        method=method,
        config=NumericalConfig(n_threads=3),
        **kwargs,
    )
    assert result.diagnostics["n_threads"] == 3


def test_concurrent_mutating_operations_are_serialized_per_model(monkeypatch):
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    call_lock = threading.Lock()
    calls = 0

    class DummyResult:
        x = np.array([0.0])
        fun = 0.0
        success = True
        nfev = 1
        message = "ok"

    def fake_minimize(*args, **kwargs):
        nonlocal calls
        with call_lock:
            calls += 1
            current = calls
        if current == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
        return DummyResult()

    monkeypatch.setattr(
        "pyscarcopula.copula.multivariate.equicorr.minimize",
        fake_minimize,
    )
    model = EquicorrGaussianCopula(d=3)
    u1 = np.random.default_rng(602).uniform(0.05, 0.95, (20, 3))
    u2 = np.random.default_rng(603).uniform(0.05, 0.95, (20, 3))

    def second_fit():
        second_attempted.set()
        return model.fit(u2, method="mle")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(model.fit, u1, method="mle")
        assert first_entered.wait(timeout=5)
        second = executor.submit(second_fit)
        assert second_attempted.wait(timeout=5)
        with call_lock:
            assert calls == 1
        release_first.set()
        assert first.result(timeout=5).success
        assert second.result(timeout=5).success
    assert calls == 2


def test_prepared_evaluator_serializes_concurrent_calls():
    rng = np.random.default_rng(604)
    u = rng.uniform(0.05, 0.95, (30, 3))
    prepared = _cpp_scar_ou.prepare_objective(
        u,
        EquicorrGaussianCopula(d=3),
        AutoTMConfig(
            transition_method="matrix",
            K=40,
            adaptive=False,
            max_K=40,
        ),
    )
    expected = prepared.neg_loglik_info(1.0, 0.1, 0.5)[0]

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(
            lambda _: prepared.neg_loglik_info(1.0, 0.1, 0.5)[0],
            range(8),
        ))

    np.testing.assert_array_equal(values, [expected] * len(values))


@pytest.mark.parametrize("d", [2, 4])
def test_student_scar_matrix_gradient_uses_internal_threads_bitwise(d):
    T = 32
    rng = np.random.default_rng(607 + d)
    u = rng.uniform(0.05, 0.95, (T, d))
    correlation = np.full((d, d), 0.15)
    np.fill_diagonal(correlation, 1.0)
    copula = StochasticStudentCopula(d=d, R=correlation)

    def evaluate(n_threads):
        prepared = _cpp_scar_ou.prepare_objective(
            u,
            copula,
            AutoTMConfig(
                transition_method="matrix",
                K=24,
                adaptive=False,
                max_K=24,
                n_threads=n_threads,
            ),
        )
        return prepared.neg_loglik_with_grad_info(1.0, 0.1, 0.5)

    expected = evaluate(1)
    module = _cpp_extension.load()
    batches_before = dict(module._parallel_runtime_info())[
        "batches_submitted"]
    actual = evaluate(4)
    batches_after = dict(module._parallel_runtime_info())[
        "batches_submitted"]

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert batches_after == batches_before + 1


def test_equicorr_scar_matrix_gradient_uses_internal_threads_bitwise():
    T, d, K = 7_000, 40, 40
    rng = np.random.default_rng(612)
    u = rng.uniform(0.05, 0.95, (T, d))
    copula = EquicorrGaussianCopula(d=d)

    def evaluate(n_threads):
        prepared = _cpp_scar_ou.prepare_objective(
            u,
            copula,
            AutoTMConfig(
                transition_method="matrix",
                K=K,
                adaptive=False,
                max_K=K,
                n_threads=n_threads,
            ),
        )
        return prepared.neg_loglik_with_grad_info(1.0, 0.1, 0.5)

    expected = evaluate(1)
    module = _cpp_extension.load()
    batches_before = dict(module._parallel_runtime_info())[
        "batches_submitted"]
    actual = evaluate(4)
    batches_after = dict(module._parallel_runtime_info())[
        "batches_submitted"]

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert batches_after == batches_before + 1


def test_equicorr_medium_scar_matrix_stays_sequential():
    T, d, K = 2_000, 40, 40
    rng = np.random.default_rng(613)
    u = rng.uniform(0.05, 0.95, (T, d))
    prepared = _cpp_scar_ou.prepare_objective(
        u,
        EquicorrGaussianCopula(d=d),
        AutoTMConfig(
            transition_method="matrix",
            K=K,
            adaptive=False,
            max_K=K,
            n_threads=8,
        ),
    )
    module = _cpp_extension.load()
    batches_before = dict(module._parallel_runtime_info())[
        "batches_submitted"]

    prepared.neg_loglik_with_grad_info(1.0, 0.1, 0.5)

    batches_after = dict(module._parallel_runtime_info())[
        "batches_submitted"]
    assert batches_after == batches_before


def test_model_lock_and_state_recover_after_failed_fit():
    d = 3
    correlation = np.full((d, d), 0.15)
    np.fill_diagonal(correlation, 1.0)
    good = np.random.default_rng(605).uniform(0.05, 0.95, (30, d))
    bad = good.copy()
    bad[0, 0] = np.nan
    config = NumericalConfig(n_threads=1)
    model = StochasticStudentCopula(d=d, R=correlation)

    with pytest.raises(ValueError):
        model.fit(bad, method="mle", config=config, maxiter=5)
    recovered = model.fit(
        good, method="mle", config=config, maxiter=5)
    fresh = StochasticStudentCopula(d=d, R=correlation).fit(
        good, method="mle", config=config, maxiter=5)

    assert recovered.copula_param == fresh.copula_param
    np.testing.assert_array_equal(
        model.log_pdf_rows(good, recovered.copula_param),
        StochasticStudentCopula(d=d, R=correlation).log_pdf_rows(
            good, fresh.copula_param),
    )


@pytest.mark.skipif(os.name == "nt", reason="fork is unavailable on Windows")
def test_forked_child_n_threads_one_ignores_inherited_pool():
    # Run in a disposable interpreter so the deliberate child-side leak of
    # inherited std::thread handles cannot affect the pytest process.
    payload = _run_clean_interpreter(
        "import json, os\n"
        "from pyscarcopula.numerical import _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "m._parallel_for_blocks_probe(32, 1, 4)\n"
        "read_fd, write_fd = os.pipe()\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.close(read_fd)\n"
        "    m._parallel_for_blocks_probe(16, 1, 1)\n"
        "    data = json.dumps(dict(m._parallel_runtime_info())).encode()\n"
        "    os.write(write_fd, data)\n"
        "    os.close(write_fd)\n"
        "    os._exit(0)\n"
        "os.close(write_fd)\n"
        "child = json.loads(os.read(read_fd, 4096).decode())\n"
        "os.close(read_fd)\n"
        "os.waitpid(pid, 0)\n"
        "print(json.dumps(child))\n"
    )
    assert payload["initialized"] is False


@pytest.mark.skipif(os.name == "nt", reason="fork is unavailable on Windows")
def test_forked_child_recreates_pool_for_parallel_work():
    payload = _run_clean_interpreter(
        "import json, os\n"
        "from pyscarcopula.numerical import _cpp_extension\n"
        "m = _cpp_extension.load()\n"
        "m._parallel_for_blocks_probe(32, 1, 4)\n"
        "parent_pid = os.getpid()\n"
        "read_fd, write_fd = os.pipe()\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.close(read_fd)\n"
        "    result = dict(m._parallel_for_blocks_probe(16, 1, 2))\n"
        "    data = json.dumps({'pid': os.getpid(), 'result': result}).encode()\n"
        "    os.write(write_fd, data)\n"
        "    os.close(write_fd)\n"
        "    os._exit(0)\n"
        "os.close(write_fd)\n"
        "child = json.loads(os.read(read_fd, 4096).decode())\n"
        "os.close(read_fd)\n"
        "os.waitpid(pid, 0)\n"
        "child['parent_pid'] = parent_pid\n"
        "print(json.dumps(child))\n"
    )
    runtime = payload["result"]["runtime"]
    assert payload["pid"] != payload["parent_pid"]
    assert runtime["owner_pid"] == payload["pid"]
    assert runtime["worker_count"] == 2
    assert runtime["batches_submitted"] == 1


@pytest.mark.skipif(
    "forkserver" not in multiprocessing.get_all_start_methods(),
    reason="forkserver is unavailable",
)
@pytest.mark.parametrize("n_threads", [1, 2])
def test_forkserver_child_uses_process_local_runtime(n_threads):
    module = _cpp_extension.load()
    module._parallel_for_blocks_probe(16, 1, 4)
    context = multiprocessing.get_context("forkserver")
    queue = context.Queue()
    process = context.Process(
        target=_runtime_child_probe, args=(queue, n_threads))
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 0
    info = queue.get(timeout=5)
    if n_threads == 1:
        assert info["initialized"] is False
    else:
        assert info["initialized"] is True
        assert info["owner_pid"] == process.pid
        assert info["worker_count"] == 2
