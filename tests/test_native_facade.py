import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyscarcopula import (
    ClaytonCopula,
    GaussianCopula,
    IndependentCopula,
    StochasticStudentCopula,
    VineCopula,
)
from pyscarcopula._native import (
    FailureContext,
    NativeError,
    NativeUnsupported,
    raise_for_status,
)
from pyscarcopula._native import _extension
from pyscarcopula._native.registry import (
    STRATEGY_REQUIREMENTS,
    descriptor_for,
    query_capability,
    strategy_support,
)
from pyscarcopula._native.threads import validate_n_threads
from pyscarcopula._native import _extension as _cpp_extension
from pyscarcopula._types import MLEResult
from pyscarcopula.api import log_likelihood, mixture_h
from pyscarcopula.strategy.mle import MLEStrategy


ROOT = Path(__file__).resolve().parents[1]


def test_native_boundary_policies_have_single_owners():
    production = tuple((ROOT / "pyscarcopula").rglob("*.py"))

    raw_import_owners = {
        path.relative_to(ROOT).as_posix()
        for path in production
        if 'import_module("pyscarcopula._native._scar_cpp")' in path.read_text(
            encoding="utf-8")
    }
    status_policy_owners = {
        path.relative_to(ROOT).as_posix()
        for path in production
        if "if status in (2, 6)" in path.read_text(encoding="utf-8")
    }
    thread_policy_owners = {
        path.relative_to(ROOT).as_posix()
        for path in production
        if "n_threads must be an integer in [1, 256]" in path.read_text(
            encoding="utf-8")
    }

    assert raw_import_owners == {"pyscarcopula/_native/_extension.py"}
    assert status_policy_owners == {"pyscarcopula/_native/errors.py"}
    assert thread_policy_owners == {"pyscarcopula/_native/threads.py"}


def test_extension_loader_and_errors_have_separate_facade_owners():
    import pyscarcopula._native as native
    from pyscarcopula._native import gas, pair, scar_ou

    assert _cpp_extension.load() is _extension.load()
    assert not hasattr(_cpp_extension, "NativeError")
    for module in (native, _cpp_extension, gas, pair, scar_ou):
        assert not hasattr(module, "available")


def test_scar_ou_fixed_draw_state_sampling_and_conditioning_contracts():
    from pyscarcopula._native import scar_ou

    stationary = scar_ou.sample_stationary_fixed_draws(
        2.0, 0.3, 4.0, np.array([0.5, -1.0, 0.25]))
    np.testing.assert_allclose(stationary, [1.3, -1.7, 0.8], atol=2e-15)

    z_grid = np.array([-1.0, 0.0, 2.0])
    probability = np.array([0.2, 0.3, 0.5])
    grid, grid_info = scar_ou.sample_state_distribution_fixed_draws(
        z_grid, probability, [0.0, 0.2, 0.499999, 0.5, 0.999], [],
        mode="grid")
    np.testing.assert_array_equal(grid, [-1.0, 0.0, 0.0, 2.0, 2.0])
    assert grid_info == {"selection_draws_used": 5, "jitter_draws_used": 0}

    histogram, histogram_info = scar_ou.sample_state_distribution_fixed_draws(
        z_grid, probability, [0.1, 0.6], [0.5, 0.25], mode="histogram")
    np.testing.assert_array_equal(histogram, [-0.75, 1.25])
    assert histogram_info == {
        "selection_draws_used": 2,
        "jitter_draws_used": 2,
    }

    conditioned_grid, conditioned_probability = scar_ou.condition_state(
        IndependentCopula(), z_grid, probability, [0.25, 0.75])
    np.testing.assert_array_equal(conditioned_grid, z_grid)
    np.testing.assert_allclose(conditioned_probability, probability, atol=0.0)

    with pytest.raises(NativeError, match="invalid_parameter"):
        scar_ou.condition_state(
            IndependentCopula(), z_grid, np.zeros(3), [0.25, 0.75])


def test_top_level_import_fails_fast_when_extension_is_missing():
    code = """
import importlib.abc
import sys

class BlockNativeExtension(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyscarcopula._native._scar_cpp':
            raise ImportError('synthetic missing native extension')
        return None

sys.meta_path.insert(0, BlockNativeExtension())
try:
    import pyscarcopula
except ImportError as exc:
    assert 'synthetic missing native extension' in str(exc)
else:
    raise AssertionError('pyscarcopula import unexpectedly succeeded')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_removed_top_level_raw_extension_path_has_no_alias():
    assert importlib.util.find_spec("pyscarcopula._scar_cpp") is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pyscarcopula._scar_cpp")


def test_exact_type_registry_builds_opaque_cpp_descriptor():
    module = _extension.load()
    model = ClaytonCopula(rotate=180)

    descriptor = descriptor_for(model)

    assert descriptor.model_id == module.NativeModelId.Clayton
    assert descriptor.dimension == 2
    assert descriptor.correlation_kind == module.CorrelationKind.NotApplicable
    assert descriptor.rotation == 180


def test_registry_covers_every_python_visible_native_model_id():
    from pyscarcopula import (
        BivariateGaussianCopula,
        EquicorrGaussianCopula,
        FrankCopula,
        GumbelCopula,
        JoeCopula,
        StudentCopula,
    )
    from pyscarcopula._native.registry import (
        registered_model_types,
        registry_entry_for,
    )

    module = _extension.load()
    models = (
        IndependentCopula(),
        ClaytonCopula(),
        FrankCopula(),
        GumbelCopula(),
        JoeCopula(),
        BivariateGaussianCopula(),
        GaussianCopula(d=3),
        StudentCopula(d=3),
        EquicorrGaussianCopula(3),
        StochasticStudentCopula(d=3),
        VineCopula(),
    )

    assert {type(model) for model in models} == set(registered_model_types())
    assert {
        descriptor_for(model).model_id.name for model in models
    } == set(module.NativeModelId.__members__)
    assert {
        registry_entry_for(model).native_id for model in models
    } == set(module.NativeModelId.__members__)
    assert set(STRATEGY_REQUIREMENTS) == {
        "MLE", "GAS", "SCAR-TM-OU", "SCAR-TM-JACOBI",
    }
    assert set(module.NativeOperation.__members__) == {
        "ParameterTransformBoundsInitialization",
        "PointDensityDerivatives",
        "RowGridDensityGradient",
        "LikelihoodObjectiveGradient",
        "StateFilterSmoother",
        "RosenblattResidual",
        "RadialGofSummary",
        "UnconditionalSamplingTransform",
        "ConditionalSamplingTransform",
        "ArbitraryConditionalMcmc",
        "EdgeStructureSelectionScore",
    }
    assert set(module.DynamicsKind.__members__) == {
        "Mle", "Gas", "ScarTmOu", "ScarTmJacobi",
    }


def test_registry_rejects_unregistered_subclasses():
    class CustomClayton(ClaytonCopula):
        pass

    with pytest.raises(NativeUnsupported, match="exact registered"):
        descriptor_for(CustomClayton())
    with pytest.raises(NativeUnsupported, match="exact registered"):
        strategy_support(CustomClayton(), "MLE")


def test_production_dispatch_rejects_subclasses_before_python_overrides():
    from pyscarcopula.strategy.predict_helpers import sample_predictive
    from pyscarcopula.vine._selection import _screen_log_likelihood

    class CustomClayton(ClaytonCopula):
        def sample_at_parameter(self, *args, **kwargs):
            raise AssertionError("custom sampling math must not execute")

        def log_likelihood(self, *args, **kwargs):
            raise AssertionError("custom likelihood math must not execute")

    model = CustomClayton()
    observations = np.array([[0.2, 0.4], [0.6, 0.8]])

    with pytest.raises(NativeUnsupported, match="exact registered"):
        sample_predictive(model, 2, np.array([0.7, 0.7]))
    with pytest.raises(NativeUnsupported, match="exact registered"):
        _screen_log_likelihood(model, observations, 0.7)


def test_likelihood_and_h_dispatch_reject_subclasses_before_overrides():
    calls = []

    class CustomClayton(ClaytonCopula):
        def log_likelihood(self, *_args, **_kwargs):
            calls.append("log_likelihood")
            return 123.0

        def h(self, *_args, **_kwargs):
            calls.append("h")
            return np.full(2, 0.25)

        def h_pair(self, *_args, **_kwargs):
            calls.append("h_pair")
            return np.full(2, 0.25), np.full(2, 0.75)

    model = CustomClayton()
    observations = np.array([[0.2, 0.4], [0.6, 0.8]])
    result = MLEResult(
        log_likelihood=0.0,
        method="MLE",
        copula_name=model.name,
        success=True,
        copula_param=0.7,
    )
    strategy = MLEStrategy()

    operations = (
        lambda: log_likelihood(model, observations, result),
        lambda: mixture_h(model, observations, result),
        lambda: strategy.log_likelihood(model, observations, result),
        lambda: strategy.rosenblatt_e2(model, observations, result),
        lambda: strategy.mixture_h_pair(model, observations, result),
        lambda: strategy.objective(model, observations, np.array([0.7])),
    )
    for operation in operations:
        with pytest.raises(NativeUnsupported, match="exact registered"):
            operation()

    assert calls == []


@pytest.mark.parametrize(
    ("model", "operation", "dynamics", "supported"),
    [
        (
            IndependentCopula(),
            "likelihood_objective_gradient",
            "MLE",
            True,
        ),
        (
            IndependentCopula(),
            "likelihood_objective_gradient",
            "GAS",
            False,
        ),
        (
            GaussianCopula(d=3),
            "conditional_sampling_transform",
            "MLE",
            True,
        ),
        (
            GaussianCopula(d=3),
            "likelihood_objective_gradient",
            "GAS",
            False,
        ),
        (
            StochasticStudentCopula(
                d=3,
                corr_mode="factor",
                factor_rank=1,
                factor_estimation="two-stage",
            ),
            "state_filter_smoother",
            "SCAR-TM-OU",
            True,
        ),
        (
            StochasticStudentCopula(
                d=3,
                corr_mode="factor",
                factor_rank=1,
                factor_estimation="joint",
            ),
            "state_filter_smoother",
            "SCAR-TM-JACOBI",
            False,
        ),
        (
            VineCopula(),
            "arbitrary_conditional_mcmc",
            "MLE",
            True,
        ),
    ],
)
def test_cpp_capability_query_covers_representative_inventory_cells(
        model, operation, dynamics, supported):
    info = query_capability(model, operation, dynamics)
    assert info.supported is supported
    assert bool(info.reason) is (not supported)


def test_strategy_requirements_are_checked_by_cpp_query():
    assert STRATEGY_REQUIREMENTS["SCAR-TM-OU"].operations == (
        "parameter_transform_bounds_initialization",
        "likelihood_objective_gradient",
        "state_filter_smoother",
    )
    assert strategy_support(ClaytonCopula(), "SCAR-TM-OU").supported
    unsupported = strategy_support(GaussianCopula(d=3), "SCAR-TM-OU")
    assert not unsupported.supported
    assert unsupported.reason


def test_gaussian_joint_factor_capability_matches_public_rejection():
    from pyscarcopula._native import _extension

    with pytest.raises(NotImplementedError, match="factor-loading score"):
        GaussianCopula(d=3, corr_mode="factor", factor_rank=1,
                       factor_estimation="joint")
    module = _extension.load()
    descriptor = module.make_typed_model_descriptor(
        module.NativeModelId.Gaussian, 3, module.CorrelationKind.Factor, 0,
        module.FactorEstimationKind.Joint)
    result = module.query_capability(
        descriptor, module.NativeOperation.LikelihoodObjectiveGradient,
        module.DynamicsKind.Mle)
    assert not result.supported
    assert "factor-loading score" in result.reason

    for model_id, estimation in (
            (module.NativeModelId.Gaussian, module.FactorEstimationKind.TwoStage),
            (module.NativeModelId.Student, module.FactorEstimationKind.Joint)):
        supported_descriptor = module.make_typed_model_descriptor(
            model_id, 3, module.CorrelationKind.Factor, 0, estimation)
        assert module.query_capability(
            supported_descriptor,
            module.NativeOperation.LikelihoodObjectiveGradient,
            module.DynamicsKind.Mle).supported


@pytest.mark.parametrize(
    ("status", "exception"),
    [
        (np.int32(1), NativeError),
        (2, ValueError),
        (3, NativeUnsupported),
        (7, FloatingPointError),
    ],
)
def test_central_status_policy(status, exception):
    with pytest.raises(exception, match="native operation failed"):
        raise_for_status(status, "operation", prefix="native")


def test_unsupported_capability_exposes_frozen_contract_fields():
    from pyscarcopula._native import ensure_capability

    model = GaussianCopula(d=3)
    with pytest.raises(NativeUnsupported) as captured:
        ensure_capability(model, "state_filter_smoother", "GAS")

    error = captured.value
    assert error.descriptor == "GaussianCopula"
    assert error.operation == "state_filter_smoother"
    assert error.dynamics == "GAS"
    assert error.reason == "static multivariate model does not support GAS"


def test_central_status_policy_preserves_structured_failure_context():
    result = {
        "status": 7,
        "failure_index": 3,
        "backend": "matrix",
        "fallback_reason": "none",
        "diagnostics": {"iterations": 4},
    }

    with pytest.raises(FloatingPointError) as captured:
        raise_for_status(result, "operation")

    error = captured.value
    assert error.status == 7
    assert error.operation == "operation"
    assert error.failure_index == 3
    assert error.diagnostics == {"iterations": 4}
    assert error.context == FailureContext(
        index=3,
        backend="matrix",
        fallback="none",
        locations={"index": 3},
        diagnostics={"iterations": 4},
    )
    assert error.failure_context is error.context


def test_objective_wrappers_propagate_native_failures(monkeypatch):
    import importlib

    from pyscarcopula import GumbelCopula, StudentCopula
    from pyscarcopula._native import static as static_likelihood
    from pyscarcopula.strategy import scar_tm
    from pyscarcopula.strategy.scar_jacobi import SCARJacobiStrategy

    observations = np.array(
        [[0.12, 0.23], [0.31, 0.45], [0.58, 0.67], [0.79, 0.86]],
        dtype=np.float64,
    )

    def unsupported(*_args, **_kwargs):
        raise NativeUnsupported("injected native objective failure")

    gas_filter = importlib.import_module("pyscarcopula.numerical.gas_filter")
    monkeypatch.setattr(
        gas_filter._cpp_gas, "negative_log_likelihood", unsupported)
    with pytest.raises(NativeUnsupported, match="native objective failure"):
        gas_filter.gas_negloglik(
            0.1, 0.2, 0.3, observations, GumbelCopula())

    monkeypatch.setattr(scar_tm._cpp_scar_ou, "neg_loglik", unsupported)
    with pytest.raises(NativeUnsupported, match="native objective failure"):
        scar_tm.SCARTMStrategy().objective(
            GumbelCopula(), observations, np.array([1.0, 0.2, 0.3]))

    def numerical_failure(*_args, **_kwargs):
        raise FloatingPointError("injected native numerical failure")

    monkeypatch.setattr(static_likelihood, "prepare_student", numerical_failure)
    student = StudentCopula(d=2, R=np.eye(2))
    with pytest.raises(FloatingPointError, match="native numerical failure"):
        student._nll_with_params(observations, np.eye(2), 6.0)

    jacobi = SCARJacobiStrategy(basis_order=3, quad_order=16)
    monkeypatch.setattr(jacobi, "_prepared_evaluator", unsupported)
    with pytest.raises(NativeUnsupported, match="native objective failure"):
        jacobi.objective(
            GumbelCopula(), observations, np.array([1.2, 0.4, 0.25]))


def test_jacobi_fit_callback_propagates_native_failure(monkeypatch):
    from pyscarcopula import GumbelCopula
    from pyscarcopula.strategy.scar_jacobi import SCARJacobiStrategy

    def unsupported(*_args, **_kwargs):
        raise NativeUnsupported("injected native Jacobi fit failure")

    class FailingEvaluator:
        neg_loglik = staticmethod(unsupported)
        neg_loglik_with_grad = staticmethod(unsupported)

    strategy = SCARJacobiStrategy(
        basis_order=3,
        quad_order=16,
        smart_init=False,
        analytical_grad=True,
    )
    monkeypatch.setattr(
        strategy, "_prepared_evaluator", lambda *_args, **_kwargs: FailingEvaluator())
    observations = np.array(
        [[0.12, 0.23], [0.31, 0.45], [0.58, 0.67], [0.79, 0.86]],
        dtype=np.float64,
    )
    with pytest.raises(NativeUnsupported, match="Jacobi fit failure"):
        strategy.fit(
            GumbelCopula(), observations,
            alpha0=np.array([1.2, 0.4, 0.25]),
            maxiter=1, maxfun=4)


@pytest.mark.parametrize("value", [1, 8, np.int64(256)])
def test_central_thread_validation_accepts_native_range(value):
    assert validate_n_threads(value) == int(value)


@pytest.mark.parametrize("value", [True, 0, 257, 1.5, "8"])
def test_central_thread_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="n_threads"):
        validate_n_threads(value)
