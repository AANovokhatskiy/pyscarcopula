"""Dynamic-vine state ownership, fit policy, and native safety regressions."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import ndtr

from pyscarcopula import BivariateGaussianCopula, VineCopula
from pyscarcopula._native import _extension, vine as native_vine
from pyscarcopula._types import (
    GASResult, LatentResult, gas_params, jacobi_params, ou_params,
)
from rvine_runtime_cases import configured_static_dvine


def _history(rows=60, dimension=3):
    z = np.random.default_rng(20260830).normal(size=(rows, dimension))
    z[:, 1] = 0.75 * z[:, 0] + np.sqrt(1.0 - 0.75**2) * z[:, 1]
    if dimension > 2:
        z[:, 2] = -0.6 * z[:, 1] + 0.8 * z[:, 2]
    return ndtr(z)


def _dynamic_vine(method):
    vine = configured_static_dvine(3)
    for index, pair in enumerate(vine.pair_copulas.values()):
        common = dict(
            method=method, copula_name=pair.copula.name,
            log_likelihood=0.0, success=True,
        )
        if method == "GAS":
            result = GASResult(
                **common, params=gas_params(0.1, 0.03, 0.7),
                scaling="unit", r_last=0.0,
            )
        elif method == "SCAR-TM-OU":
            result = LatentResult(
                **common, params=ou_params(2.0, 0.8 - index * 0.6, 0.4),
                K=32, grid_range=5.0, transition_method="local",
            )
        else:
            result = LatentResult(
                **common, params=jacobi_params(2.0, 0.7 - index * 0.2, 0.4),
                K=32, transition_method="local", gh_order=7,
            )
        pair.fit_result = result
        pair.param = None
    vine._last_u = _history()
    vine._last_u.flags.writeable = False
    return vine


@pytest.mark.parametrize("method", ["GAS", "SCAR-TM-OU", "SCAR-TM-JACOBI"])
@pytest.mark.parametrize("horizon", ["current", "next"])
@pytest.mark.parametrize("dynamic_conditioning", ["ignore", "given_only"])
def test_suffix_rebuild_does_not_reuse_another_edges_predictive_state(
        method, horizon, dynamic_conditioning):
    vine = _dynamic_vine(method)
    given = {0: 0.8, 1: 0.2}
    suffix = vine._suffix_sampling_state(given)
    assert suffix is not None and suffix[2] != vine._edge_map
    options = dict(
        horizon=horizon, dynamic_conditioning=dynamic_conditioning)

    expected = vine.predict(
        128, given=given, rng=np.random.default_rng(123), **options)
    vine._predict_history_cache.clear()
    vine.predict(16, rng=np.random.default_rng(456), **options)
    actual = vine.predict(
        128, given=given, rng=np.random.default_rng(123), **options)
    np.testing.assert_array_equal(actual, expected)

    # The remapped query must not poison later queries in the original order.
    actual_unconditional = vine.predict(
        128, rng=np.random.default_rng(321), **options)
    vine._predict_history_cache.clear()
    expected_unconditional = vine.predict(
        128, rng=np.random.default_rng(321), **options)
    np.testing.assert_array_equal(actual_unconditional, expected_unconditional)


@pytest.mark.parametrize("policy", ["fallback", "keep", "raise"])
@pytest.mark.parametrize("search", [None, "beam", "multi-start"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_auto_vine_honors_dynamic_failure_policy(policy, search, dimension):
    vine = VineCopula(candidates=[BivariateGaussianCopula])
    options = {} if search is None else dict(
        given_vars=[0], structure_search=search)
    if dimension == 3:
        # Keep upper-tree edges Gaussian so independence selection cannot
        # bypass the dynamic fit whose failure policy is under test.
        options["copulas"] = [
            [(BivariateGaussianCopula, 0)] * 2,
            [(BivariateGaussianCopula, 0)],
        ]
    if policy == "raise":
        with pytest.raises(RuntimeError, match="dynamic fit failed"):
            vine.fit(_history(dimension=dimension), method="gas", maxiter=1,
                     dynamic_failure_policy=policy, **options)
        assert vine.pair_copulas is None
        return
    vine.fit(_history(dimension=dimension), method="gas", maxiter=1,
             dynamic_failure_policy=policy, **options)
    for pair in vine.pair_copulas.values():
        assert pair.fit_diagnostics["attempted_success"] is False
        assert pair.fit_diagnostics["dynamic_failure_policy"] == policy
        assert pair.fit_result.method == ("GAS" if policy == "keep" else "MLE")
    assert vine.fit_result.success is all(
        pair.fit_result.success for pair in vine.pair_copulas.values())
    if policy == "keep":
        assert vine.fit_result.success is False


@pytest.mark.parametrize("search", [None, "beam", "multi-start"])
def test_failed_auto_refit_preserves_previous_model(search):
    observations = _history()
    vine = VineCopula(candidates=[BivariateGaussianCopula])
    vine.fit(observations, method="mle")
    previous_result = vine.fit_result
    previous_pairs = vine.pair_copulas
    previous_likelihood = vine.log_likelihood(observations)
    previous_samples = vine.predict(32, rng=np.random.default_rng(123))
    options = {} if search is None else dict(
        given_vars=[0], structure_search=search)

    with pytest.raises(RuntimeError, match="dynamic fit failed"):
        vine.fit(observations, method="gas", maxiter=1,
                 dynamic_failure_policy="raise", **options)

    assert vine.fit_result is previous_result
    assert vine.pair_copulas is previous_pairs
    assert vine.log_likelihood(observations) == previous_likelihood
    np.testing.assert_array_equal(
        vine.predict(32, rng=np.random.default_rng(123)), previous_samples)


def test_auto_vine_rejects_invalid_policy_before_selection(monkeypatch):
    from pyscarcopula.vine import vine as runtime

    def unexpected_selection(*args, **kwargs):
        pytest.fail("invalid fit policy reached structure selection")

    monkeypatch.setattr(runtime, "select_rvine_structure", unexpected_selection)
    with pytest.raises(ValueError, match="dynamic_failure_policy"):
        VineCopula().fit(_history(), method="gas", dynamic_failure_policy="invalid")


def _native_dynamic_request(method="GAS"):
    vine = _dynamic_vine(method)
    observations = _history(rows=5)
    module = _extension.load()
    keys = native_vine.density_active_keys(vine._trees, vine._edge_map)
    edges, pack = native_vine.compile_dynamic_rosenblatt_edges(
        module, vine.pair_copulas, keys, len(observations))
    plan = native_vine.compile_density_plan(
        module, vine.d, vine._trees, vine._edge_map, keys,
        residual_node_keys=native_vine.rosenblatt_residual_node_keys(vine.matrix))
    return module, plan, edges, pack, observations


@pytest.mark.parametrize("method", ["GAS", "SCAR-TM-OU", "SCAR-TM-JACOBI"])
@pytest.mark.parametrize("value", [-1.0, 2.0, np.nan, np.inf, -np.inf])
def test_dynamic_native_rejects_invalid_observations(method, value):
    module, plan, edges, pack, observations = _native_dynamic_request(method)
    observations[2, 1] = value
    result = module.dynamic_rvine_rosenblatt_transform(
        plan, edges, pack.scalar_parameters, pack.row_parameters, observations, 1)
    assert result["status"] == 6
    assert result["failure_row"] == 2
    assert np.asarray(result["residuals"]).size == 0


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_dynamic_native_preserves_closed_unit_boundary_clipping(value):
    module, plan, edges, pack, observations = _native_dynamic_request()
    observations[2, 1] = value
    result = module.dynamic_rvine_rosenblatt_transform(
        plan, edges, pack.scalar_parameters, pack.row_parameters, observations, 1)
    assert result["status"] == 0
    assert np.all(np.isfinite(result["residuals"]))


@pytest.mark.parametrize("defect", ["missing_residuals", "short_residuals", "missing_outputs"])
def test_dynamic_native_rejects_incomplete_rosenblatt_plan_in_subprocess(defect):
    # A regression of this boundary previously crashed the interpreter. Keep
    # the reproducer isolated even after adding the validation guard.
    source = """
import ctypes
import sys
if sys.platform == 'win32':
    ctypes.windll.kernel32.SetErrorMode(3)
sys.path.insert(0, 'tests')
from test_dynamic_vine_regressions import _native_dynamic_request
module, original, edges, pack, observations = _native_dynamic_request()
plan = module.RVineDensityPlan()
for name in dir(original):
    if not name.startswith('_') and name != 'residual_nodes':
        setattr(plan, name, getattr(original, name))
defect = sys.argv[1]
if defect == 'short_residuals':
    plan.residual_nodes = original.residual_nodes[:1]
elif defect == 'missing_outputs':
    plan.residual_nodes = original.input_nodes
    first, second = list(plan.output1_nodes), list(plan.output2_nodes)
    first[-1] = second[-1] = -1
    plan.output1_nodes, plan.output2_nodes = first, second
result = module.dynamic_rvine_rosenblatt_transform(
    plan, edges, pack.scalar_parameters, pack.row_parameters, observations, 1)
assert result['status'] == 2, result
assert len(result['residuals']) == 0
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", source, defect],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
