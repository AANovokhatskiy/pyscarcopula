"""End-to-end thread ownership for non-vine multivariate sampling."""

import copy

import numpy as np
import pytest

from pyscarcopula import (
    EquicorrGaussianCopula,
    GaussianCopula,
    StochasticStudentCopula,
    StudentCopula,
    api,
)
from pyscarcopula._native import _extension, multivariate as native
from pyscarcopula._types import (
    GASResult, LatentResult, MLEResult, gas_params, ou_params,
)


_MODES = ("fixed", "shrinkage", "cholesky", "factor")
_CASES = (
    [("equicorr", "fixed", method) for method in ("mle", "gas", "scar-tm-ou")]
    + [("stochastic", mode, method)
       for mode in _MODES for method in ("mle", "gas", "scar-tm-ou")
       if (mode, method) != ("cholesky", "gas")]
    + [(family, mode, "mle")
       for family in ("gaussian", "student") for mode in _MODES]
)


@pytest.fixture(scope="module", params=_CASES, ids=lambda case: "-".join(case))
def fitted_model(request):
    family, mode, method = request.param
    d = 4
    correlation = .25 * np.ones((d, d)) + .75 * np.eye(d)
    observations = np.random.default_rng(31).uniform(.08, .92, (40, d))
    if family == "equicorr":
        model = EquicorrGaussianCopula(d)
    else:
        cls = {"stochastic": StochasticStudentCopula,
               "gaussian": GaussianCopula, "student": StudentCopula}[family]
        options = {"corr_mode": mode}
        if mode == "factor":
            options.update(factor_rank=1, factor_loadings=np.full((d, 1), .45))
        else:
            options["R"] = correlation
        model = cls(d, **options)
    if family in ("gaussian", "student") or (family, method) == ("equicorr", "mle"):
        assert model.fit(observations, method="mle").success
    else:
        # Fixed, valid strategy states isolate sampling from optimizer convergence.
        fields = dict(copula_name=model.name, success=True, log_likelihood=0.)
        if method == "mle":
            result = MLEResult(method="MLE", copula_param=6., **fields)
        elif method == "gas":
            result = GASResult(
                method="GAS", params=gas_params(.15, .03, .65),
                scaling="unit", r_last=.2 if family == "equicorr" else 6.,
                **fields)
        else:
            result = LatentResult(
                method="SCAR-TM-OU", params=ou_params(2., .4, .8),
                K=40, adaptive=False, grid_method="dense", grid_range=7.,
                transition_method="matrix", **fields)
        model.fit_result = result
    model._last_u = observations.copy()
    return model


def _draw(model, operation, n, given, rng, **kwargs):
    if operation.startswith("api_"):
        return getattr(api, operation[4:])(
            model, model._last_u, model.fit_result, n,
            given=given, rng=rng, **kwargs)
    if operation == "sample":
        return model.sample(n, rng=rng, **kwargs)
    if operation in ("sample_at_parameter", "sample_at_parameter_batches"):
        kwargs["r"] = .2 if isinstance(model, EquicorrGaussianCopula) else 6.
    else:
        kwargs["given"] = given
    if operation.endswith("batches"):
        blocks = list(getattr(model, operation)(
            n, batch_rows=128, rng=rng, **kwargs))
        return np.concatenate(blocks) if blocks else np.empty((0, model.dimension))
    return getattr(model, operation)(n, rng=rng, **kwargs)


def _operations(model):
    operations = [
        "sample", "sample_conditional", "predict", "api_sample", "api_predict"]
    operations.extend(name for name in (
        "sample_batches", "predict_batches", "sample_at_parameter",
        "sample_at_parameter_batches") if hasattr(model, name))
    return operations


@pytest.mark.parametrize("given", [None, {}, {0: .2}], ids=["none", "empty", "partial"])
def test_sampling_threads_reach_native_and_preserve_seed(
        fitted_model, given, monkeypatch):
    """Inspect native result diagnostics, not just Python keyword forwarding."""
    model = fitted_model
    native_results = []
    original = native._sampling_values

    def capture(result, module, operation):
        native_results.append({**dict(result), "operation": operation})
        return original(result, module, operation)

    monkeypatch.setattr(native, "_sampling_values", capture)
    runtime = _extension.load()
    for operation in _operations(model):
        before = runtime._parallel_runtime_info()["batches_submitted"]
        expected = _draw(
            model, operation, 257, given, np.random.default_rng(127), n_threads=1)
        assert (
            runtime._parallel_runtime_info()["batches_submitted"] == before
        ), operation
        assert native_results and all(
            info["n_threads_requested"] == 1 for info in native_results)
        native_results.clear()
        actual = _draw(
            model, operation, 257, given, np.random.default_rng(127), n_threads=4)
        np.testing.assert_array_equal(actual, expected, err_msg=operation)
        assert native_results and all(
            info["n_threads_requested"] == 4 for info in native_results
        ), operation
        recursive = model.fit_result.method == "GAS" and (
            operation in ("sample", "sample_batches", "api_sample")
            or (operation == "sample_conditional" and not given))
        dense_conditional = all(
            info["operation"].startswith((
                "dense Gaussian conditional", "dense Student conditional"))
            for info in native_results)
        # The dense conditional policy needs at least 65536 row*d*d work.
        if recursive or dense_conditional:
            assert all(
                info["parallel_blocks"] == 1 for info in native_results
            ), operation
        else:
            assert any(
                info["parallel_blocks"] > 1 for info in native_results
            ), operation
            assert (
                runtime._parallel_runtime_info()["batches_submitted"] > before
            ), operation
        native_results.clear()


def test_large_implicit_conditional_sampling_runs_native_workers(
        fitted_model, monkeypatch):
    diagnostics = []
    original = native._sampling_values

    def capture(result, module, operation):
        diagnostics.append(dict(result))
        return original(result, module, operation)

    monkeypatch.setattr(native, "_sampling_values", capture)
    model = fitted_model
    expected = model.sample_conditional(
        4097, given={0: .2}, rng=np.random.default_rng(131), n_threads=1)
    diagnostics.clear()
    before = _extension.load()._parallel_runtime_info()["batches_submitted"]
    actual = model.sample_conditional(
        4097, given={0: .2}, rng=np.random.default_rng(131), n_threads=4)
    np.testing.assert_array_equal(actual, expected)
    assert diagnostics and all(info["n_threads_requested"] == 4 for info in diagnostics)
    assert any(info["parallel_blocks"] == 4 for info in diagnostics)
    assert _extension.load()._parallel_runtime_info()["batches_submitted"] > before


@pytest.mark.parametrize("given", [None, {}, {0: .2}, {j: .2 for j in range(4)}],
                         ids=["none", "empty", "partial", "all"])
@pytest.mark.parametrize(
    "n_threads", [0, -1, 257, True, np.bool_(True), 1.0, 1.5, None])
def test_invalid_sampling_threads_fail_without_advancing_rng(
        fitted_model, given, n_threads):
    for operation in _operations(fitted_model):
        rng = np.random.default_rng(128)
        previous = copy.deepcopy(rng.bit_generator.state)
        with pytest.raises(ValueError, match="n_threads"):
            _draw(fitted_model, operation, 8, given, rng, n_threads=n_threads)
        assert rng.bit_generator.state == previous, operation


@pytest.mark.parametrize("n", [0, 1])
def test_zero_and_one_sampling_threads(fitted_model, n):
    for operation in _operations(fitted_model):
        for given in (None, {0: .2}, {j: .2 for j in range(4)}):
            # GAS model reproduction and strategy APIs require a positive count.
            dense = getattr(fitted_model, "corr_mode", None) != "factor"
            positive_only = fitted_model.fit_result.method == "GAS" and (
                operation.startswith("api_") or (dense and (
                    operation == "sample"
                    or (operation == "sample_conditional" and not given))))
            if n == 0 and positive_only:
                with pytest.raises(ValueError, match="n must be positive"):
                    _draw(
                        fitted_model, operation, n, given,
                        np.random.default_rng(129), n_threads=4)
                continue
            actual = _draw(
                fitted_model, operation, n, given,
                np.random.default_rng(129), n_threads=4)
            assert actual.shape == (n, fitted_model.dimension), operation
            assert np.isfinite(actual).all(), operation


def test_sampling_defaults_ignore_thread_environment(fitted_model, monkeypatch):
    monkeypatch.setenv("PYSCARCOPULA_NUM_THREADS", "8")
    for operation in _operations(fitted_model):
        implicit = _draw(
            fitted_model, operation, 17, {0: .2}, np.random.default_rng(130))
        explicit = _draw(
            fitted_model, operation, 17, {0: .2}, np.random.default_rng(130),
            n_threads=np.int64(1))
        np.testing.assert_array_equal(implicit, explicit, err_msg=operation)
