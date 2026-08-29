"""FV-4 runtime proofs for the mandatory native numerical boundary."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy import linalg as scipy_linalg
from scipy import special as scipy_special
from scipy import stats as scipy_stats

from rvine_runtime_cases import configured_static_dvine

from pyscarcopula import (
    ClaytonCopula,
    GaussianCopula,
    StudentCopula,
)
from pyscarcopula._native import _extension
from pyscarcopula._native import gas as gas_native
from pyscarcopula._native import jacobi as jacobi_native
from pyscarcopula._native import model_policy
from pyscarcopula._native import multivariate as multivariate_native
from pyscarcopula._native import pair as pair_native
from pyscarcopula._native import registry as native_registry
from pyscarcopula._native import scar_ou as scar_ou_native
from pyscarcopula._native import static as static_native
from pyscarcopula._native import statistics as statistics_native
from pyscarcopula._native.errors import NativeUnsupported
from pyscarcopula.copula.multivariate import EquicorrGaussianCopula
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
from pyscarcopula.stattests import cvm_test, gof_test
from pyscarcopula.strategy.gas import GASStrategy
from pyscarcopula.strategy.mle import MLEStrategy
from pyscarcopula.strategy.multivariate_mle import (
    StaticMLEEvaluation,
    StaticMLEProblem,
    run_static_multivariate_mle,
)
from pyscarcopula.strategy.predict_helpers import conditional_sample_bivariate
from pyscarcopula.strategy.scar_jacobi import SCARJacobiStrategy
from pyscarcopula.strategy.scar_tm import SCARTMStrategy

import pyscarcopula.strategy.gas as gas_strategy_module
import pyscarcopula.strategy.mle as mle_strategy_module
import pyscarcopula.strategy.multivariate_mle as multivariate_mle_module
import pyscarcopula.strategy.scar_jacobi as jacobi_strategy_module
import pyscarcopula.strategy.scar_tm as scar_tm_strategy_module


_CORRELATION = np.array([
    [1.0, 0.35, -0.15],
    [0.35, 1.0, 0.25],
    [-0.15, 0.25, 1.0],
])
_OU_CONFIG = AutoTMConfig(
    transition_method="matrix",
    K=16,
    adaptive=False,
    max_K=16,
)


def _observations(rows=12, dimension=2, seed=20260829):
    return np.random.default_rng(seed).uniform(
        0.05, 0.95, size=(rows, dimension))


def _dependent_observations(rows=30):
    rng = np.random.default_rng(20260830)
    common = rng.uniform(0.1, 0.9, size=rows)
    noise = rng.uniform(-0.04, 0.04, size=(rows, 3))
    return np.clip(common[:, None] + noise, 0.02, 0.98)


def _pair_transform_trace():
    copula = ClaytonCopula()
    return lambda: copula.transform(np.array([-0.2, 0.3]))


def _pair_inverse_transform_trace():
    copula = ClaytonCopula()
    return lambda: copula.inv_transform(np.array([0.8, 1.2]))


def _pair_dtransform_trace():
    copula = ClaytonCopula()
    return lambda: copula.dtransform(np.array([-0.2, 0.3]))


def _pair_tau_to_parameter_trace():
    copula = ClaytonCopula()
    return lambda: copula.tau_to_param(np.array([0.2, 0.4]))


def _pair_parameter_to_tau_trace():
    copula = ClaytonCopula()
    return lambda: copula.param_to_tau(np.array([0.8, 1.2]))


def _pair_bounds_trace():
    copula = ClaytonCopula()
    return lambda: model_policy.public_bounds(copula)


def _pair_initialization_trace():
    copula = ClaytonCopula()
    return lambda: model_policy.default_pair_mle_parameter(copula)


def _pair_pdf_trace():
    copula = ClaytonCopula()
    return lambda: copula.pdf([0.2, 0.4], [0.7, 0.6], 1.2)


def _pair_log_pdf_trace():
    copula = ClaytonCopula()
    return lambda: copula.log_pdf([0.2, 0.4], [0.7, 0.6], 1.2)


def _pair_density_derivative_trace():
    copula = ClaytonCopula()
    return lambda: copula.dlog_pdf_dr([0.2, 0.4], [0.7, 0.6], 1.2)


def _pair_h_trace():
    copula = ClaytonCopula()
    return lambda: copula.h([0.2, 0.4], [0.7, 0.6], 1.2)


def _pair_h_pair_trace():
    copula = ClaytonCopula()
    return lambda: copula.h_pair([0.2, 0.4], [0.7, 0.6], 1.2)


def _pair_h_inverse_trace():
    copula = ClaytonCopula()
    return lambda: copula.h_inverse([0.2, 0.4], [0.7, 0.6], 1.2)


def _pair_pdf_grid_trace():
    copula = ClaytonCopula()
    observations = _observations(4)
    return lambda: copula.copula_grid_batch(
        observations, np.array([-0.4, 0.0, 0.4]))


def _pair_pdf_and_gradient_grid_trace():
    copula = ClaytonCopula()
    observations = _observations(4)
    return lambda: copula.pdf_and_grad_on_grid_batch(
        observations, np.array([-0.4, 0.0, 0.4]))


def _static_trace():
    copula = ClaytonCopula()
    observations = _observations(6)
    return lambda: static_native.prepare(copula, observations)


def _gaussian_likelihood_trace():
    copula = _gaussian_model()
    observations = _observations(8, 3, seed=20260853)
    return lambda: copula.log_likelihood(observations)


def _student_likelihood_trace():
    copula = _student_model()
    observations = _observations(8, 3, seed=20260854)
    return lambda: copula.log_likelihood(observations)


def _gas_trace():
    copula = ClaytonCopula()
    observations = _observations(8)
    return lambda: gas_native.negative_log_likelihood_and_gradient(
        0.02, 0.08, 0.7, observations, copula)


def _scar_ou_trace():
    copula = ClaytonCopula()
    observations = _observations(8)
    return lambda: scar_ou_native.neg_loglik_with_grad(
        1.2, 0.1, 0.7, observations, copula, _OU_CONFIG)


def _scar_jacobi_trace():
    copula = ClaytonCopula()
    observations = _observations(8)
    return lambda: jacobi_native.PreparedScarJacobiEvaluator(
        observations,
        copula,
        basis_order=4,
        quad_order=16,
        gh_order=3,
        transition_method="local_fixed",
    )


def _gaussian_model():
    copula = GaussianCopula(d=3, R=_CORRELATION)
    copula.corr = _CORRELATION.copy()
    return copula


def _student_model():
    copula = StudentCopula(d=3, R=_CORRELATION)
    copula.shape = _CORRELATION.copy()
    copula.df = 6.0
    return copula


def _gaussian_rosenblatt_trace():
    copula = _gaussian_model()
    observations = _observations(8, 3, seed=20260839)
    return lambda: gof_test(copula, observations, to_pobs=False)


def _student_rosenblatt_trace():
    copula = _student_model()
    observations = _observations(8, 3, seed=20260840)
    return lambda: gof_test(copula, observations, to_pobs=False)


def _vine_rosenblatt_trace():
    vine = configured_static_dvine(4)
    observations = _observations(8, 4, seed=20260841)
    return lambda: gof_test(vine, observations, to_pobs=False)


def _radial_gof_trace():
    residuals = _observations(8, 3, seed=20260842)
    return lambda: cvm_test(residuals)


def _pair_sampling_trace():
    copula = ClaytonCopula()
    return lambda: copula.sample_at_parameter(
        3, 1.2, rng=np.random.default_rng(20260843))


def _gaussian_sampling_trace():
    copula = _gaussian_model()
    return lambda: copula.sample(3, rng=np.random.default_rng(20260831))


def _student_sampling_trace():
    copula = _student_model()
    return lambda: copula.sample(3, rng=np.random.default_rng(20260844))


def _vine_sampling_trace():
    vine = configured_static_dvine(4)
    return lambda: vine.sample(3, rng=np.random.default_rng(20260845))


def _pair_conditional_sampling_trace():
    copula = ClaytonCopula()
    return lambda: conditional_sample_bivariate(
        copula,
        3,
        1.2,
        given={0: 0.4},
        rng=np.random.default_rng(20260846),
    )


def _gaussian_conditional_sampling_trace():
    copula = _gaussian_model()
    return lambda: copula.sample_conditional(
        3, {1: 0.6}, rng=np.random.default_rng(20260847))


def _student_conditional_sampling_trace():
    copula = _student_model()
    return lambda: copula.sample_conditional(
        3, {1: 0.6}, rng=np.random.default_rng(20260848))


def _vine_suffix_conditional_sampling_trace():
    vine = configured_static_dvine(4)
    given = None
    for variable in range(vine.d):
        candidate = {variable: 0.6}
        if vine._suffix_sampling_state(candidate) is not None:
            given = candidate
            break
    assert given is not None
    return lambda: vine.predict(
        3, given=given, rng=np.random.default_rng(20260849))


def _vine_arbitrary_conditional_mcmc_trace():
    vine = configured_static_dvine(4)
    peel_order = [
        int(vine.matrix[vine.d - 1 - column, column])
        for column in range(vine.d)
    ]
    given = {peel_order[0]: 0.25, peel_order[2]: 0.75}
    assert vine._suffix_sampling_state(given) is None
    return lambda: vine.predict(
        2,
        given=given,
        mcmc_steps=1,
        mcmc_burnin=0,
        rng=np.random.default_rng(20260850),
    )


def _vine_density_trace():
    observations = _observations(8, 4, seed=20260851)
    vine = configured_static_dvine(4)
    return lambda: vine.log_likelihood(observations)


def _kendall_selection_trace():
    observations = _dependent_observations()
    return lambda: statistics_native.kendall_tau(
        observations[:, 0], observations[:, 1])


def _candidate_score_trace():
    return lambda: statistics_native.candidate_score(
        -12.0, 1, 30, "aic")


@dataclass(frozen=True)
class _RuntimeOperationCase:
    name: str
    capability: str
    raw_symbol: str
    operation_factory: object


_RUNTIME_OPERATION_CASES = (
    _RuntimeOperationCase(
        "pair-transform", "parameter_transform_bounds_initialization",
        "copula_transform", _pair_transform_trace),
    _RuntimeOperationCase(
        "pair-inverse-transform", "parameter_transform_bounds_initialization",
        "copula_inverse_transform", _pair_inverse_transform_trace),
    _RuntimeOperationCase(
        "pair-transform-derivative",
        "parameter_transform_bounds_initialization",
        "copula_dtransform", _pair_dtransform_trace),
    _RuntimeOperationCase(
        "pair-tau-to-parameter", "parameter_transform_bounds_initialization",
        "copula_tau_to_param", _pair_tau_to_parameter_trace),
    _RuntimeOperationCase(
        "pair-parameter-to-tau", "parameter_transform_bounds_initialization",
        "copula_param_to_tau", _pair_parameter_to_tau_trace),
    _RuntimeOperationCase(
        "pair-public-bounds", "parameter_transform_bounds_initialization",
        "model_public_parameter_bounds", _pair_bounds_trace),
    _RuntimeOperationCase(
        "pair-default-initialization",
        "parameter_transform_bounds_initialization",
        "model_default_pair_mle_parameter", _pair_initialization_trace),
    _RuntimeOperationCase(
        "pair-pdf", "point_density_derivatives",
        "copula_pdf", _pair_pdf_trace),
    _RuntimeOperationCase(
        "pair-log-pdf", "point_density_derivatives",
        "copula_log_pdf", _pair_log_pdf_trace),
    _RuntimeOperationCase(
        "pair-density-derivative", "point_density_derivatives",
        "copula_dlog_pdf_dr", _pair_density_derivative_trace),
    _RuntimeOperationCase(
        "pair-h", "point_density_derivatives",
        "copula_h", _pair_h_trace),
    _RuntimeOperationCase(
        "pair-h-pair", "point_density_derivatives",
        "copula_h_pair", _pair_h_pair_trace),
    _RuntimeOperationCase(
        "pair-inverse-h", "point_density_derivatives",
        "copula_h_inverse", _pair_h_inverse_trace),
    _RuntimeOperationCase(
        "pair-grid", "row_grid_density_gradient",
        "copula_pdf_grid", _pair_pdf_grid_trace),
    _RuntimeOperationCase(
        "pair-grid-gradient", "row_grid_density_gradient",
        "copula_pdf_and_grad_grid", _pair_pdf_and_gradient_grid_trace),
    _RuntimeOperationCase(
        "static-objective", "likelihood_objective_gradient",
        "StaticCopulaEvaluator", _static_trace),
    _RuntimeOperationCase(
        "gaussian-likelihood", "likelihood_objective_gradient",
        "StaticCopulaEvaluator", _gaussian_likelihood_trace),
    _RuntimeOperationCase(
        "student-likelihood", "likelihood_objective_gradient",
        "StaticCopulaEvaluator", _student_likelihood_trace),
    _RuntimeOperationCase(
        "vine-density", "likelihood_objective_gradient",
        "rvine_log_pdf_rows", _vine_density_trace),
    _RuntimeOperationCase(
        "gas-state", "state_filter_smoother",
        "GasEvaluator", _gas_trace),
    _RuntimeOperationCase(
        "scar-tm-ou-state", "state_filter_smoother",
        "ScarOuEvaluator", _scar_ou_trace),
    _RuntimeOperationCase(
        "scar-tm-jacobi-state", "state_filter_smoother",
        "PreparedScarJacobiEvaluator", _scar_jacobi_trace),
    _RuntimeOperationCase(
        "gaussian-rosenblatt", "rosenblatt_residual",
        "dense_gaussian_rosenblatt_transform", _gaussian_rosenblatt_trace),
    _RuntimeOperationCase(
        "student-rosenblatt", "rosenblatt_residual",
        "dense_student_rosenblatt_transform", _student_rosenblatt_trace),
    _RuntimeOperationCase(
        "vine-rosenblatt", "rosenblatt_residual",
        "rvine_rosenblatt_transform", _vine_rosenblatt_trace),
    _RuntimeOperationCase(
        "radial-gof", "radial_gof_summary",
        "radial_uniform_summary", _radial_gof_trace),
    _RuntimeOperationCase(
        "pair-sampling", "unconditional_sampling_transform",
        "copula_sample_from_uniforms", _pair_sampling_trace),
    _RuntimeOperationCase(
        "gaussian-sampling", "unconditional_sampling_transform",
        "multivariate_gaussian_sample_from_normals", _gaussian_sampling_trace),
    _RuntimeOperationCase(
        "student-sampling", "unconditional_sampling_transform",
        "multivariate_student_sample_from_normal_uniforms",
        _student_sampling_trace),
    _RuntimeOperationCase(
        "vine-sampling", "unconditional_sampling_transform",
        "rvine_sample", _vine_sampling_trace),
    _RuntimeOperationCase(
        "pair-conditional-sampling", "conditional_sampling_transform",
        "copula_conditional_sample_from_uniforms",
        _pair_conditional_sampling_trace),
    _RuntimeOperationCase(
        "gaussian-conditional-sampling", "conditional_sampling_transform",
        "multivariate_gaussian_conditional_from_uniforms",
        _gaussian_conditional_sampling_trace),
    _RuntimeOperationCase(
        "student-conditional-sampling", "conditional_sampling_transform",
        "multivariate_student_conditional_from_normal_uniforms",
        _student_conditional_sampling_trace),
    _RuntimeOperationCase(
        "vine-suffix-conditional-sampling", "conditional_sampling_transform",
        "rvine_conditional_sample", _vine_suffix_conditional_sampling_trace),
    _RuntimeOperationCase(
        "vine-arbitrary-conditional-mcmc", "arbitrary_conditional_mcmc",
        "rvine_mcmc_chunk", _vine_arbitrary_conditional_mcmc_trace),
    _RuntimeOperationCase(
        "selection-kendall", "edge_structure_selection_score",
        "statistics_kendall_tau", _kendall_selection_trace),
    _RuntimeOperationCase(
        "selection-candidate-score", "edge_structure_selection_score",
        "statistics_information_criterion", _candidate_score_trace),
)


def test_runtime_operation_manifest_covers_every_native_capability():
    names = [case.name for case in _RUNTIME_OPERATION_CASES]
    assert len(names) == len(set(names))
    assert {case.capability for case in _RUNTIME_OPERATION_CASES} == set(
        native_registry._OPERATION_NAMES)


@pytest.mark.parametrize(
    "case",
    _RUNTIME_OPERATION_CASES,
    ids=lambda case: case.name,
)
def test_public_operation_reaches_raw_native_entry_once_without_fallback(
    monkeypatch, case,
):
    operation = case.operation_factory()
    module = _extension.load()
    calls = []

    def fail_native(*args, **kwargs):
        calls.append((args, kwargs))
        raise NativeUnsupported(f"FV-4 sentinel at {case.name}")

    monkeypatch.setattr(module, case.raw_symbol, fail_native)
    with pytest.raises(
            NativeUnsupported, match=f"FV-4 sentinel at {case.name}"):
        operation()
    assert len(calls) == 1


def _install_forbidden_numerical_sentinels(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden Python numerical owner was called")

    for owner, names in (
        (
            np.linalg,
            (
                "cholesky", "det", "eigvals", "eigvalsh", "eigh", "inv",
                "lstsq", "norm", "pinv", "slogdet", "solve", "svd",
            ),
        ),
        (
            scipy_linalg,
            ("cholesky", "det", "eigh", "inv", "lu", "solve", "svd"),
        ),
        (
            scipy_special,
            (
                "betainc", "betaincinv", "gammaln", "ndtr", "ndtri",
                "roots_hermite", "roots_hermitenorm", "roots_jacobi",
                "stdtrit",
            ),
        ),
    ):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)

    for distribution, names in (
        (scipy_stats.norm, ("cdf", "logpdf", "pdf", "ppf")),
        (scipy_stats.t, ("cdf", "logpdf", "pdf", "ppf")),
        (scipy_stats.multivariate_normal, ("logpdf", "pdf")),
        (scipy_stats.multivariate_t, ("logpdf", "pdf")),
    ):
        for name in names:
            monkeypatch.setattr(distribution, name, forbidden)


def test_supported_model_runtime_survives_forbidden_python_numerical_owners(
    monkeypatch,
):
    _install_forbidden_numerical_sentinels(monkeypatch)

    pair = ClaytonCopula()
    gaussian = _gaussian_model()
    student = _student_model()
    pair_observations = _observations(8)
    multi_observations = _observations(8, 3, seed=20260832)
    vine_observations = _observations(8, 4, seed=20260852)
    vine = configured_static_dvine(4)

    operations = (
        lambda: pair.pdf([0.2, 0.4], [0.7, 0.6], 1.2),
        lambda: pair.h([0.2, 0.4], [0.7, 0.6], 1.2),
        lambda: pair.h_inverse([0.2, 0.4], [0.7, 0.6], 1.2),
        lambda: pair.pdf_and_grad_on_grid_batch(
            pair_observations, np.array([-0.4, 0.0, 0.4])),
        lambda: static_native.prepare(
            pair, pair_observations).objective_and_gradient(1.2),
        lambda: gas_native.negative_log_likelihood_and_gradient(
            0.02, 0.08, 0.7, pair_observations, pair),
        lambda: scar_ou_native.neg_loglik_with_grad(
            1.2, 0.1, 0.7, pair_observations, pair, _OU_CONFIG),
        lambda: jacobi_native.PreparedScarJacobiEvaluator(
            pair_observations,
            pair,
            basis_order=4,
            quad_order=16,
            gh_order=3,
            transition_method="local_fixed",
        ).neg_loglik_with_grad(1.2, 0.4, 0.25),
        lambda: gaussian.log_likelihood(multi_observations),
        lambda: gaussian.sample(4, rng=np.random.default_rng(20260833)),
        lambda: gaussian.sample_conditional(
            4, {1: 0.6}, rng=np.random.default_rng(20260834)),
        lambda: student.log_likelihood(multi_observations),
        lambda: student.sample(4, rng=np.random.default_rng(20260835)),
        lambda: student.sample_conditional(
            4, {1: 0.6}, rng=np.random.default_rng(20260836)),
        lambda: vine.log_likelihood(vine_observations),
        lambda: vine.sample(4, rng=np.random.default_rng(20260837)),
    )
    for operation in operations:
        operation()


class _OptimizerProbeComplete(RuntimeError):
    pass


def _run_one_optimizer_callback(fun, x0, *args, **kwargs):
    fun(np.asarray(x0, dtype=np.float64))
    raise _OptimizerProbeComplete


def test_mle_optimizer_callback_makes_one_typed_native_evaluator_call(
    monkeypatch,
):
    calls = []

    class Evaluator:
        def objective_and_gradient(self, parameter, **kwargs):
            calls.append((parameter, kwargs))
            return 1.0, np.zeros(1, dtype=np.float64)

    monkeypatch.setattr(static_native, "prepare", lambda *args, **kwargs: Evaluator())
    monkeypatch.setattr(
        mle_strategy_module, "minimize", _run_one_optimizer_callback)
    _install_forbidden_numerical_sentinels(monkeypatch)

    with pytest.raises(_OptimizerProbeComplete):
        MLEStrategy().fit(
            ClaytonCopula(), _observations(8), alpha0=1.2)

    assert len(calls) == 1


def test_gas_optimizer_callback_makes_one_typed_native_evaluator_call(
    monkeypatch,
):
    calls = []

    def objective(*args, **kwargs):
        calls.append((args, kwargs))
        return 1.0, np.zeros(3, dtype=np.float64)

    monkeypatch.setattr(
        gas_native, "negative_log_likelihood_and_gradient", objective)
    monkeypatch.setattr(
        gas_strategy_module, "minimize", _run_one_optimizer_callback)
    _install_forbidden_numerical_sentinels(monkeypatch)

    with pytest.raises(_OptimizerProbeComplete):
        GASStrategy().fit(
            ClaytonCopula(),
            _observations(8),
            gamma0=np.array([0.02, 0.08, 0.7]),
        )

    assert len(calls) == 1


def test_multivariate_optimizer_callback_makes_one_typed_evaluator_call(
    monkeypatch,
):
    calls = []
    callback_call_counts = []

    def evaluate(parameters):
        calls.append(np.asarray(parameters, dtype=np.float64).copy())
        return StaticMLEEvaluation(
            objective=1.0,
            gradient=np.zeros(len(parameters), dtype=np.float64),
        )

    def run_one_callback(fun, x0, *args, **kwargs):
        before = len(calls)
        fun(np.asarray(x0, dtype=np.float64))
        callback_call_counts.append(len(calls) - before)
        raise _OptimizerProbeComplete

    monkeypatch.setattr(
        multivariate_mle_module, "minimize", run_one_callback)
    _install_forbidden_numerical_sentinels(monkeypatch)

    problem = StaticMLEProblem(
        family="FV-4",
        initial_parameters=np.array([5.0]),
        bounds=((2.001, None),),
        evaluate=evaluate,
    )
    with pytest.raises(_OptimizerProbeComplete):
        run_static_multivariate_mle(
            problem,
            optimizer_options={"gtol": 1e-4},
            fail_value=1e10,
        )

    assert len(calls) == 2
    assert callback_call_counts == [1]


def test_scar_ou_optimizer_callback_makes_one_typed_native_evaluator_call(
    monkeypatch,
):
    calls = []

    class Evaluator:
        def update_copula(self, copula):
            return None

        def neg_loglik_with_grad_info(self, kappa, mu, nu):
            calls.append((kappa, mu, nu))
            return 1.0, np.zeros(3, dtype=np.float64), {}

    monkeypatch.setattr(
        scar_ou_native,
        "prepare_objective",
        lambda *args, **kwargs: Evaluator(),
    )
    monkeypatch.setattr(
        scar_tm_strategy_module, "minimize", _run_one_optimizer_callback)
    _install_forbidden_numerical_sentinels(monkeypatch)

    with pytest.raises(_OptimizerProbeComplete):
        SCARTMStrategy(
            K=16,
            adaptive=False,
            max_K=16,
            transition_method="matrix",
            analytical_grad=True,
            smart_init=False,
        ).fit(
            ClaytonCopula(),
            _observations(8),
            alpha0=np.array([1.2, 0.1, 0.7]),
        )

    assert len(calls) == 1


def test_scar_jacobi_optimizer_callback_makes_one_typed_native_call(
    monkeypatch,
):
    calls = []

    class Evaluator:
        def neg_loglik_with_grad(self, kappa, m, xi):
            calls.append((kappa, m, xi))
            return 1.0, np.zeros(3, dtype=np.float64)

    monkeypatch.setattr(
        SCARJacobiStrategy,
        "_prepared_evaluator",
        lambda self, u, copula: Evaluator(),
    )
    monkeypatch.setattr(
        jacobi_strategy_module, "minimize", _run_one_optimizer_callback)
    _install_forbidden_numerical_sentinels(monkeypatch)

    with pytest.raises(_OptimizerProbeComplete):
        SCARJacobiStrategy(
            basis_order=4,
            quad_order=16,
            gh_order=3,
            transition_method="local_fixed",
            analytical_grad=True,
            smart_init=False,
        ).fit(
            ClaytonCopula(),
            _observations(8),
            alpha0=np.array([1.2, 0.4, 0.25]),
        )

    assert len(calls) == 1


def test_unsupported_descriptor_stops_before_optimizer_and_numerical_entry(
    monkeypatch,
):
    model = GaussianCopula(d=3, R=_CORRELATION)
    observations = _observations(8, 3, seed=20260838)
    numerical_calls = []

    def forbidden_numerical(*args, **kwargs):
        numerical_calls.append((args, kwargs))
        raise AssertionError("unsupported descriptor reached numerical code")

    monkeypatch.setattr(_extension.load(), "GasEvaluator", forbidden_numerical)
    monkeypatch.setattr(
        "pyscarcopula.strategy.gas.minimize", forbidden_numerical)

    with pytest.raises(NativeUnsupported, match=r"C\+\+.*GAS supports"):
        GASStrategy().fit(
            model,
            observations,
            gamma0=np.array([0.02, 0.08, 0.7]),
        )
    assert numerical_calls == []


class _UniformRng:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def uniform(self, low, high, size):
        self.calls.append((low, high, size))
        return self.values


class _NormalRng:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def standard_normal(self, size):
        self.calls.append(size)
        return self.values


class _StudentRng:
    def __init__(self, normals, uniforms):
        self.normals = normals
        self.uniforms = uniforms
        self.calls = []

    def uniform(self, low, high, size):
        self.calls.append(("uniform", low, high, size))
        return self.uniforms

    def standard_normal(self, size):
        self.calls.append(("standard_normal", size))
        return self.normals


def test_pair_rng_boundary_forwards_uniforms_without_python_transform(
    monkeypatch,
):
    draws = np.arange(6, dtype=np.float64).reshape(3, 2) / 7.0
    output = np.full((3, 2), 0.25)
    captured = {}

    def fixed_draw_call(copula, uniforms, parameter):
        captured.update(
            copula=copula, uniforms=uniforms, parameter=parameter)
        return output

    monkeypatch.setattr(pair_native, "sample_from_uniforms", fixed_draw_call)
    copula = ClaytonCopula()
    rng = _UniformRng(draws)

    result = copula.sample_at_parameter(3, 1.2, rng=rng)

    assert result is output
    assert captured["copula"] is copula
    assert captured["uniforms"] is draws
    np.testing.assert_array_equal(captured["parameter"], np.full(3, 1.2))
    assert rng.calls == [(0, 1, (3, 2))]


def test_gaussian_rng_boundary_forwards_normals_without_python_transform(
    monkeypatch,
):
    draws = np.arange(9, dtype=np.float64).reshape(3, 3) / 10.0
    output = np.full((3, 3), 0.5)
    captured = {}

    def fixed_draw_call(correlation, normals, *, n_threads):
        captured.update(
            correlation=correlation, normals=normals, n_threads=n_threads)
        return output

    monkeypatch.setattr(
        multivariate_native,
        "gaussian_sample_from_normals",
        fixed_draw_call,
    )
    copula = GaussianCopula(d=3, R=_CORRELATION)
    copula.corr = _CORRELATION.copy()
    rng = _NormalRng(draws)

    result = copula.sample(3, rng=rng)

    assert result is output
    assert captured["normals"] is draws
    assert captured["n_threads"] == 1
    np.testing.assert_array_equal(captured["correlation"], _CORRELATION)
    assert rng.calls == [(3, 3)]


def test_student_rng_boundary_forwards_raw_draws_without_python_transform(
    monkeypatch,
):
    normals = np.arange(9, dtype=np.float64).reshape(3, 3) / 10.0
    uniforms = np.array([0.2, 0.4, 0.8])
    output = np.full((3, 3), 0.75)
    captured = {}

    def fixed_draw_call(correlation, df, normal_draws, chi_square_uniforms):
        captured.update(
            correlation=correlation,
            df=df,
            normal_draws=normal_draws,
            chi_square_uniforms=chi_square_uniforms,
        )
        return output

    monkeypatch.setattr(
        multivariate_native,
        "student_sample_from_normal_uniforms",
        fixed_draw_call,
    )
    copula = StudentCopula(d=3, R=_CORRELATION)
    copula.shape = _CORRELATION.copy()
    copula.df = 6.0
    rng = _StudentRng(normals, uniforms)

    result = copula.sample(3, rng=rng)

    assert result is output
    assert captured["normal_draws"] is normals
    assert captured["chi_square_uniforms"] is uniforms
    assert captured["df"] == 6.0
    np.testing.assert_array_equal(captured["correlation"], _CORRELATION)
    assert rng.calls == [
        ("uniform", 0.0, 1.0, 3),
        ("standard_normal", (3, 3)),
    ]


def test_equicorr_rng_boundary_uses_native_draw_count_and_raw_normals(
    monkeypatch,
):
    row_draws = np.arange(12, dtype=np.float64).reshape(3, 4) / 13.0
    common_draws = np.array([0.25, -0.5])
    output = np.full((3, 4), 0.6)
    captured = {}

    class EquicorrRng:
        def __init__(self):
            self.calls = []

        def standard_normal(self, size):
            self.calls.append(size)
            return row_draws if size == (3, 4) else common_draws

    def common_count(parameters, dimension, n_rows):
        captured["planner"] = (parameters.copy(), dimension, n_rows)
        return 2

    def fixed_draw_call(parameters, dimension, normal, common):
        captured["fixed"] = (parameters, dimension, normal, common)
        return output

    monkeypatch.setattr(
        multivariate_native,
        "equicorr_gaussian_common_draw_count",
        common_count,
    )
    monkeypatch.setattr(
        multivariate_native,
        "equicorr_gaussian_sample_from_normals",
        fixed_draw_call,
    )
    copula = EquicorrGaussianCopula(4)
    rng = EquicorrRng()

    result = copula.sample_at_parameter(3, 0.25, rng=rng)

    assert result is output
    planned_parameters, planned_dimension, planned_rows = captured["planner"]
    np.testing.assert_array_equal(planned_parameters, np.full(3, 0.25))
    assert (planned_dimension, planned_rows) == (4, 3)
    parameters, dimension, normal, common = captured["fixed"]
    np.testing.assert_array_equal(parameters, planned_parameters)
    assert dimension == 4
    assert normal is row_draws
    assert common is common_draws
    assert rng.calls == [(3, 4), 2]
