"""Regression tests for RVine public-boundary validation and memory bounds."""

from contextlib import nullcontext

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    LBFGSBConfig,
    NumericalConfig,
    RVineCopula,
    VineCopula,
)
from pyscarcopula._native import static as static_likelihood
from pyscarcopula._native.errors import NativeError
from pyscarcopula.stattests import rvine_rosenblatt_transform
from pyscarcopula.vine._rvine_dissmann import select_rvine


def _independent_vine():
    data = np.array([
        [0.15, 0.25],
        [0.35, 0.45],
        [0.55, 0.65],
        [0.75, 0.85],
    ])
    return RVineCopula(candidates=[IndependentCopula]).fit(data)


def _gaussian_vine():
    rng = np.random.default_rng(12)
    first = rng.standard_normal(250)
    second = 0.7 * first + np.sqrt(1.0 - 0.7 ** 2) * rng.standard_normal(250)
    order_first = np.argsort(np.argsort(first)) + 1
    order_second = np.argsort(np.argsort(second)) + 1
    data = np.column_stack((order_first, order_second)) / 251.0
    return RVineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(
        data,
        copulas=[[(BivariateGaussianCopula, 0)]],
    )


@pytest.mark.parametrize(
    "bad_data",
    [
        np.array([[0.2, np.nan], [0.4, 0.5]]),
        np.array([[0.2, np.inf], [0.4, 0.5]]),
        np.array([[-0.1, 0.2], [0.4, 0.5]]),
        np.array([[1.1, 0.2], [0.4, 0.5]]),
        np.array([[0.0, 0.2], [0.4, 0.5]]),
        np.array([[1.0, 0.2], [0.4, 0.5]]),
        np.empty((0, 2)),
    ],
)
def test_fit_rejects_invalid_pseudo_observations_before_selection(bad_data):
    with pytest.raises(ValueError):
        RVineCopula(candidates=[IndependentCopula]).fit(bad_data)


def test_fit_rejects_complex_data_without_lossy_cast():
    data = np.array([
        [0.2 + 1.0j, 0.3],
        [0.4, 0.5],
    ])
    with pytest.raises(TypeError, match="real values"):
        RVineCopula(candidates=[IndependentCopula]).fit(data)


@pytest.mark.parametrize(
    "factory",
    [
        lambda a: a.tolist(),
        lambda a: a.astype(np.float32),
        lambda a: np.asfortranarray(a),
        lambda a: a[:, ::-1],
        lambda a: a.astype(object),
    ],
)
def test_fit_accepts_supported_array_representations_without_mutation(factory):
    base = np.array([
        [0.15, 0.25],
        [0.35, 0.45],
        [0.55, 0.65],
        [0.75, 0.85],
    ])
    data = factory(base)
    if isinstance(data, np.ndarray):
        before = data.copy()
        data.flags.writeable = False

    vine = RVineCopula(candidates=[IndependentCopula]).fit(data)

    assert vine.d == 2
    if isinstance(data, np.ndarray):
        np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize(
    "bad_data",
    [
        0.5,
        np.array([0.2, 0.4]),
        np.zeros((2, 2, 1)),
        np.array([[0, 1], [1, 0]], dtype=np.int64),
    ],
)
def test_fit_rejects_ambiguous_shapes_and_integer_pseudo_observations(
        bad_data):
    with pytest.raises((TypeError, ValueError)):
        RVineCopula(candidates=[IndependentCopula]).fit(bad_data)


@pytest.mark.parametrize(
    "bad_data",
    [
        np.array([[0.2, np.nan]]),
        np.array([[0.2, np.inf]]),
        np.array([[-0.1, 0.2]]),
        np.array([[1.1, 0.2]]),
        np.empty((0, 2)),
    ],
)
def test_explicit_likelihood_rejects_invalid_pseudo_observations(bad_data):
    with pytest.raises(ValueError):
        _independent_vine().log_likelihood(bad_data)


def test_predict_rejects_invalid_explicit_history():
    vine = _independent_vine()
    with pytest.raises(ValueError, match="finite"):
        vine.predict(4, u=np.array([[0.2, np.nan]]))
    with pytest.raises(ValueError, match="pseudo-observations"):
        vine.predict(4, u=np.array([[-0.1, 0.2]]))


def test_direct_selector_rejects_invalid_data():
    with pytest.raises(ValueError, match="finite"):
        select_rvine(np.array([[0.2, np.inf], [0.4, 0.5]]))
    with pytest.raises(ValueError, match="at least one row"):
        select_rvine(np.empty((0, 2)))


def test_static_sampling_batches_without_changing_seeded_output():
    vine = _independent_vine()
    expected = vine.sample(
        31,
        rng=np.random.default_rng(20260726),
        batch_rows=31,
    )
    actual = vine.sample(
        31,
        rng=np.random.default_rng(20260726),
        batch_rows=7,
    )
    np.testing.assert_array_equal(actual, expected)


def test_static_sampling_passes_scalar_edge_parameter_paths(monkeypatch):
    vine = _gaussian_vine()
    original = vine._sample_with_r
    observed = []

    def recording_sample(n, r_all, rng, **kwargs):
        observed.append((
            n,
            tuple(np.asarray(value).size for value in r_all.values()),
        ))
        return original(n, r_all, rng, **kwargs)

    monkeypatch.setattr(vine, "_sample_with_r", recording_sample)
    samples = vine.sample(
        17,
        rng=np.random.default_rng(3),
        batch_rows=6,
    )

    assert samples.shape == (17, 2)
    assert [rows for rows, _ in observed] == [6, 6, 5]
    assert all(sizes == (1,) for _, sizes in observed)


def test_sample_memory_budget_is_checked_before_allocation():
    vine = _independent_vine()
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        vine.sample(
            100,
            batch_rows=10,
            memory_budget_bytes=100,
            rng=np.random.default_rng(4),
        )


def test_sample_overflow_scale_request_is_rejected_before_allocation():
    with pytest.raises(MemoryError, match="memory_budget_bytes"):
        _independent_vine().sample(
            2**62,
            batch_rows=8,
            memory_budget_bytes=1,
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_sample_rejects_invalid_batch_rows(value):
    vine = _independent_vine()
    with pytest.raises((TypeError, ValueError)):
        vine.sample(10, batch_rows=value)


@pytest.mark.parametrize("stage", ["screening", "refinement"])
@pytest.mark.parametrize("error_type", [NativeError, MemoryError, RuntimeError, ValueError])
def test_family_selection_propagates_errors_and_preserves_fitted_state(
        monkeypatch, stage, error_type):
    vine = _gaussian_vine()
    original_result = vine.fit_result
    original_pairs = vine.pair_copulas
    expected = vine.sample(11, rng=np.random.default_rng(71))

    def fail(*args, **kwargs):
        raise error_type("injected candidate failure")

    target = (
        "pyscarcopula._native.static.prepare"
        if stage == "screening" else
        "pyscarcopula.vine._selection._fit_mle_direct")
    monkeypatch.setattr(target, fail)
    with pytest.raises(error_type, match="injected candidate failure"):
        vine.fit(vine._last_u)

    assert vine.fit_result is original_result
    assert vine.pair_copulas is original_pairs
    np.testing.assert_array_equal(
        vine.sample(11, rng=np.random.default_rng(71)), expected)


@pytest.mark.parametrize("stage", ["screening", "refinement"])
def test_family_selection_does_not_report_independence_when_all_candidates_fail(
        monkeypatch, stage):
    vine = _gaussian_vine()

    def fail(*args, **kwargs):
        raise FloatingPointError("injected numerical failure")

    target = (
        "pyscarcopula.vine._selection._screen_log_likelihood"
        if stage == "screening" else
        "pyscarcopula.vine._selection._fit_mle_direct")
    monkeypatch.setattr(target, fail)
    with pytest.warns(RuntimeWarning, match="injected numerical failure"):
        with pytest.raises(FloatingPointError, match="No candidate family"):
            vine.fit(vine._last_u)


@pytest.mark.parametrize("fixed_families", [False, True])
@pytest.mark.parametrize("failure_recovers", [False, True])
def test_native_numerical_failure_cannot_publish_penalty_as_fitted_model(
        monkeypatch, fixed_families, failure_recovers):
    vine = _gaussian_vine()
    original_result = vine.fit_result
    original_pairs = vine.pair_copulas
    expected = vine.sample(11, rng=np.random.default_rng(71))
    evaluator_type = static_likelihood.StaticLikelihoodEvaluator
    original_evaluate = evaluator_type.result
    calls = []

    def fail(self, parameter):
        calls.append(parameter)
        if failure_recovers and len(calls) > 1:
            return original_evaluate(self, parameter)
        return dict(
            status=7, negative_log_likelihood=np.inf,
            negative_gradient=0.0, failure_index=0)

    monkeypatch.setattr(evaluator_type, "result", fail)
    fit_kwargs = (
        {"copulas": [[(BivariateGaussianCopula, 0)]]}
        if fixed_families else {})
    warning = (
        nullcontext() if fixed_families else
        pytest.warns(RuntimeWarning, match="refinement failed"))
    with warning, pytest.raises(FloatingPointError):
        vine.fit(vine._last_u, **fit_kwargs)

    assert len(calls) == 2  # optimizer trial, then strict final validation
    assert vine.fit_result is original_result
    assert vine.pair_copulas is original_pairs
    np.testing.assert_array_equal(
        vine.sample(11, rng=np.random.default_rng(71)), expected)


@pytest.mark.parametrize("kind", ["cvine", "dvine", "rvine"])
@pytest.mark.parametrize("seed", [872, 1919])
@pytest.mark.parametrize("sign", [1, -1])
def test_auto_family_fit_accepts_near_independent_itau_starts(kind, seed, sign):
    data = np.random.default_rng(seed).uniform(0.01, 0.99, (400, 2))
    if sign < 0:
        data[:, 1] = 1.0 - data[:, 1]
    vine = (
        RVineCopula() if kind == "rvine" else
        getattr(VineCopula, kind)(2))

    vine.fit(data)

    assert vine.fit_result.success
    assert np.isfinite(vine.log_likelihood())
    assert vine.log_likelihood(data) == pytest.approx(vine.log_likelihood())


@pytest.mark.parametrize("kind", ["cvine", "dvine", "rvine"])
@pytest.mark.parametrize("sign", [1, -1])
def test_auto_gaussian_fit_projects_near_perfect_itau_starts(kind, sign):
    data = np.column_stack((np.arange(1, 51), np.arange(1, 51))) / 51.0
    data[[24, 25], 1] = data[[25, 24], 1]
    if sign < 0:
        data[:, 1] = 1.0 - data[:, 1]
    options = {"candidates": [BivariateGaussianCopula]}
    vine = (
        RVineCopula(**options) if kind == "rvine" else
        getattr(VineCopula, kind)(2, **options))

    vine.fit(data)

    assert vine.fit_result.success
    edge = vine.pair_copulas[(0, 0)]
    assert type(edge.copula) is BivariateGaussianCopula
    assert edge.fit_result.copula_param == pytest.approx(sign * 0.9999)
    assert vine.log_likelihood() == pytest.approx(222.60627414091616)


@pytest.mark.parametrize("kind", ["cvine", "dvine", "rvine"])
@pytest.mark.parametrize("n", [2, 5, 50])
@pytest.mark.parametrize("sign", [1, -1])
def test_auto_gaussian_fit_handles_exact_kendall_endpoints(kind, n, sign):
    ranks = np.arange(1, n + 1) / (n + 1.0)
    data = np.column_stack((ranks, ranks if sign > 0 else 1.0 - ranks))
    options = {"candidates": [BivariateGaussianCopula]}
    vine = (
        RVineCopula(**options) if kind == "rvine" else
        getattr(VineCopula, kind)(2, **options))

    vine.fit(data)

    edge = vine.pair_copulas[(0, 0)]
    assert type(edge.copula) is BivariateGaussianCopula
    assert vine.fit_result.success
    rho = sign * 0.9999
    assert edge.fit_result.copula_param == rho
    z = norm.ppf(data)
    expected = np.sum(
        multivariate_normal.logpdf(z, cov=[[1.0, rho], [rho, 1.0]])
        - norm.logpdf(z).sum(axis=1))
    assert vine.log_likelihood() == pytest.approx(expected, rel=0.0, abs=2e-7)
    assert expected > 1.0  # Gaussian beats the independence AIC baseline.


@pytest.mark.parametrize("kind", ["cvine", "dvine", "rvine"])
@pytest.mark.parametrize("n", [2, 5, 50])
@pytest.mark.parametrize("sign", [1, -1])
def test_default_pool_retains_bounded_candidate_at_exact_tau(kind, n, sign):
    ranks = np.arange(1, n + 1) / (n + 1.0)
    data = np.column_stack((ranks, ranks if sign > 0 else 1.0 - ranks))
    vine = RVineCopula() if kind == "rvine" else getattr(VineCopula, kind)(2)

    with pytest.warns(RuntimeWarning, match="exact Kendall limits require finite bounds"):
        vine.fit(data)

    assert type(vine.pair_copulas[(0, 0)].copula) is BivariateGaussianCopula
    assert vine.fit_result.success
    assert vine.log_likelihood() > 1.0
    sample = norm.ppf(vine.predict(500, rng=np.random.default_rng(841)))
    assert sign * np.corrcoef(sample, rowvar=False)[0, 1] > 0.99


@pytest.mark.parametrize("family", [ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula])
def test_unbounded_exact_tau_refit_fails_and_preserves_previous_model(family):
    original_data = _gaussian_vine()._last_u
    vine = VineCopula.dvine(2, candidates=[family]).fit(original_data)
    previous_result = vine.fit_result
    previous_pairs = vine.pair_copulas
    expected_sample = vine.sample(11, rng=np.random.default_rng(71))
    ranks = np.arange(1, 51) / 51.0

    with pytest.warns(RuntimeWarning, match="exact Kendall limits require finite bounds"):
        with pytest.raises(FloatingPointError, match="No candidate family"):
            vine.fit(np.column_stack((ranks, ranks)))

    assert vine.fit_result is previous_result
    assert vine.pair_copulas is previous_pairs
    np.testing.assert_array_equal(
        vine.sample(11, rng=np.random.default_rng(71)), expected_sample)


@pytest.mark.parametrize("family", [ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula])
def test_bounded_archimedean_pool_handles_exact_tau(family):
    ranks = np.arange(1, 51) / 51.0
    vine = VineCopula.dvine(
        2, candidates=[family], transform_type="logistic",
    ).fit(np.column_stack((ranks, ranks)))
    edge = vine.pair_copulas[(0, 0)]

    assert type(edge.copula) is family
    assert vine.fit_result.success
    assert edge.fit_result.copula_param == family(transform_type="logistic").bounds[0][1]
    assert vine.log_likelihood() > 1.0
    assert np.isfinite(vine.predict(100, rng=np.random.default_rng(841))).all()


@pytest.mark.parametrize(
    "family, alpha0",
    [(ClaytonCopula, 0.0), (FrankCopula, 0.0), (GumbelCopula, 1.0),
     (JoeCopula, 1.0), (BivariateGaussianCopula, 0.99999)],
)
def test_auto_family_fit_does_not_project_explicit_invalid_alpha0(family, alpha0):
    data = _gaussian_vine()._last_u
    with pytest.raises(ValueError, match="alpha0.*outside copula bounds"):
        VineCopula.dvine(2, candidates=[family]).fit(data, alpha0=alpha0)


@pytest.mark.parametrize("stage", ["screening", "native_refinement"])
def test_family_selection_reports_local_numerical_failure_and_uses_valid_family(
        monkeypatch, stage):
    from pyscarcopula.vine import _selection

    data = _gaussian_vine()._last_u
    original = _selection._screen_log_likelihood

    def screen(copula, *args, **kwargs):
        if type(copula) is ClaytonCopula and stage == "screening":
            raise FloatingPointError("injected Clayton failure")
        value, evaluator = original(copula, *args, **kwargs)
        if type(copula) is ClaytonCopula and stage == "native_refinement":
            evaluator.result = lambda parameter: dict(
                status=7, negative_log_likelihood=np.inf,
                negative_gradient=0.0, failure_index=0)
        return value, evaluator

    monkeypatch.setattr(_selection, "_screen_log_likelihood", screen)
    with pytest.warns(RuntimeWarning, match="Clayton.*failed"):
        vine = RVineCopula.dvine(
            2, candidates=[ClaytonCopula, BivariateGaussianCopula],
        ).fit(data)
    assert type(vine.pair_copulas[(0, 0)].copula) is BivariateGaussianCopula
    assert vine.log_likelihood() > 0


@pytest.mark.parametrize("fixed_structure", [False, True])
@pytest.mark.parametrize("overrides, expected_maxiter", [({}, 17), ({"maxiter": 1}, 1)])
def test_automatic_families_forward_optimizer_config_and_native_threads(
        monkeypatch, fixed_structure, overrides, expected_maxiter):
    from pyscarcopula._native import static
    from pyscarcopula.strategy import mle

    data = _gaussian_vine()._last_u
    original_minimize = mle.minimize
    original_prepare = static.prepare
    calls = []
    threads = []

    def minimize(fun, x0, **kwargs):
        calls.append((np.asarray(x0).copy(), kwargs["options"]))
        return original_minimize(fun, x0, **kwargs)

    def prepare(*args, **kwargs):
        threads.append(kwargs.get("n_threads", 1))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(mle, "minimize", minimize)
    monkeypatch.setattr(static, "prepare", prepare)
    vine = (
        RVineCopula.dvine(2, candidates=[BivariateGaussianCopula])
        if fixed_structure else
        RVineCopula(candidates=[BivariateGaussianCopula]))
    vine.fit(
        data, alpha0=[0.4], **overrides,
        config=NumericalConfig(
            n_threads=2,
            mle_optimizer=LBFGSBConfig(maxiter=17, gtol=0.2, ftol=2e-6)))

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [0.4])
    assert calls[0][1]["maxiter"] == expected_maxiter
    assert calls[0][1]["gtol"] == 0.2
    assert calls[0][1]["ftol"] == 2e-6
    assert threads == [2]


def test_automatic_families_reject_ambiguous_alpha0_before_screening(monkeypatch):
    data = _gaussian_vine()._last_u

    def unexpected(*args, **kwargs):
        pytest.fail("ambiguous alpha0 reached native screening")

    monkeypatch.setattr("pyscarcopula._native.static.prepare", unexpected)
    with pytest.raises(ValueError, match="exactly one"):
        RVineCopula.dvine(
            2, candidates=[BivariateGaussianCopula, ClaytonCopula],
        ).fit(data, alpha0=[0.5])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.1, 1.1])
def test_rosenblatt_rejects_invalid_observations_before_native(monkeypatch, value):
    vine = _independent_vine()

    def unexpected(*args, **kwargs):
        pytest.fail("invalid observations reached native Rosenblatt")

    monkeypatch.setattr(
        "pyscarcopula.stattests._rvine_rosenblatt_transform_native", unexpected)
    with pytest.raises(ValueError):
        rvine_rosenblatt_transform(vine, np.full((3, 2), value))


@pytest.mark.parametrize("dtype", [complex, object])
def test_rosenblatt_rejects_complex_values_without_cast_warning(dtype):
    with pytest.raises(TypeError, match="real values"):
        rvine_rosenblatt_transform(
            _independent_vine(), np.full((3, 2), 0.5 + 1j, dtype=dtype))


def test_rosenblatt_preserves_endpoint_clipping_and_readonly_input():
    vine = _independent_vine()
    observations = np.array([[0., 1.], [.2, .7]])
    before = observations.copy()
    observations.flags.writeable = False
    actual = rvine_rosenblatt_transform(vine, observations)
    np.testing.assert_array_equal(actual, np.clip(before, 1e-6, 1-1e-6))
    np.testing.assert_array_equal(observations, before)


@pytest.mark.parametrize("given", [None, {0: 0.6}])
def test_static_predict_needs_no_training_trace_or_full_parameter_paths(
        monkeypatch, given):
    import pyscarcopula.vine.vine as vine_module

    vine = _gaussian_vine()
    expected = vine.predict(19, given=given, rng=np.random.default_rng(37))
    original = vine_module._edge_r_for_predict

    def unexpected(*args, **kwargs):
        pytest.fail("static prediction requested a training trace")

    def predict_parameter(edge, n, **kwargs):
        assert n == 1
        return original(edge, n, **kwargs)

    monkeypatch.setattr(vine, "_compute_pseudo_obs", unexpected)
    monkeypatch.setattr(vine_module, "_edge_r_for_predict", predict_parameter)
    monkeypatch.setattr(vine_module, "_DEFAULT_STATIC_SAMPLE_BATCH_ROWS", 7)
    actual, diagnostics = vine.predict(
        19, given=given, rng=np.random.default_rng(37),
        return_diagnostics=True)
    np.testing.assert_array_equal(actual, expected)
    assert diagnostics["conditional_method"] == (
        "unconditional" if given is None else "suffix")
    assert not vine._predict_history_cache
