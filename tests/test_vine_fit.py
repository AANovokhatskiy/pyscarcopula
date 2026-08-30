"""Permanent tests for fitting pair edges on a fixed vine structure."""

import numpy as np
import pytest
from scipy.stats import norm

import pyscarcopula.vine._rvine_dissmann as dissmann_module
import pyscarcopula.vine._structure as structure_module
import pyscarcopula.vine._vine_fit as vine_fit_module
from pyscarcopula import (
    BivariateGaussianCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    VineCopula,
)
from pyscarcopula._types import GASResult, gas_params
from pyscarcopula.api import fit as fit_copula
from pyscarcopula.vine import cvine_structure, dvine_structure
from pyscarcopula.vine._rvine_dissmann import (
    VineStructureSelection,
    select_rvine,
    select_rvine_structure,
)
from pyscarcopula.vine._vine_fit import VineEdgeFit, fit_vine_edges


def _gaussian_data(rows=240, dimension=4, seed=20260726):
    correlation = np.fromfunction(
        lambda i, j: 0.65 ** np.abs(i - j),
        (dimension, dimension),
    )
    rng = np.random.default_rng(seed)
    return norm.cdf(rng.multivariate_normal(
        np.zeros(dimension), correlation, size=rows))


def _fixed_specs(trees, copula_class=BivariateGaussianCopula, rotation=0):
    return [
        [(copula_class, rotation) for _ in level]
        for level in trees
    ]


def _manual_gaussian_log_likelihood(u, trees):
    pseudo = {
        (variable, frozenset()): u[:, variable].copy()
        for variable in range(u.shape[1])
    }
    total = 0.0
    for tree, level in enumerate(trees):
        for conditioned, conditioning in level:
            var1, var2 = sorted(conditioned)
            first = pseudo[(var1, conditioning)]
            second = pseudo[(var2, conditioning)]
            pair = np.column_stack((first, second))
            copula = BivariateGaussianCopula()
            result = fit_copula(copula, pair, method="mle")
            total += result.log_likelihood

            if tree < u.shape[1] - 2:
                parameter = np.full(len(u), result.copula_param)
                pseudo[(var2, conditioning | {var1})] = copula.h(
                    second, first, parameter)
                pseudo[(var1, conditioning | {var2})] = copula.h(
                    first, second, parameter)
    return float(total)


@pytest.mark.parametrize(
    "structure_factory",
    [cvine_structure, dvine_structure],
)
def test_fixed_c_and_d_vine_match_manual_pair_decomposition(
        structure_factory):
    u = _gaussian_data(dimension=3)
    trees = structure_factory(3).to_trees()

    fitted = fit_vine_edges(
        u,
        trees,
        copulas=_fixed_specs(trees),
        method="mle",
    )

    assert fitted.log_likelihood == pytest.approx(
        _manual_gaussian_log_likelihood(u, trees),
        rel=1e-10,
        abs=1e-10,
    )


def test_fixed_fit_does_not_call_dissmann_or_mst(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("structure selection must not run")

    monkeypatch.setattr(
        dissmann_module, "_build_tree_0", unexpected_call)
    monkeypatch.setattr(
        structure_module, "_maximum_spanning_tree", unexpected_call)
    trees = dvine_structure(4).to_trees()

    fitted = fit_vine_edges(
        _gaussian_data(dimension=4),
        trees,
        copulas=_fixed_specs(trees, IndependentCopula),
    )

    assert fitted.log_likelihood == 0.0
    assert fitted.parameter_count == 0


def test_auto_selected_trees_refit_with_same_families_reproduce_edges():
    u = _gaussian_data(dimension=4)
    trees, auto_levels = select_rvine(
        u,
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
        method="mle",
    )
    specs = [
        [
            (type(pair.copula), int(pair.copula.rotate))
            for pair in level
        ]
        for level in auto_levels
    ]

    fixed = fit_vine_edges(
        u,
        trees,
        copulas=specs,
        method="mle",
    )

    for tree, level in enumerate(auto_levels):
        for edge, auto_pair in enumerate(level):
            fixed_pair = fixed.pair_copulas[(tree, edge)]
            assert type(fixed_pair.copula) is type(auto_pair.copula)
            assert fixed_pair.copula.rotate == auto_pair.copula.rotate
            assert fixed_pair.param == pytest.approx(
                auto_pair.param, abs=1e-6)
            assert fixed_pair.log_likelihood == pytest.approx(
                auto_pair.log_likelihood, abs=2e-6)


def test_structure_selection_result_excludes_temporary_fitted_edges():
    selection = select_rvine_structure(
        _gaussian_data(dimension=4),
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    )

    assert isinstance(selection, VineStructureSelection)
    assert selection.structure == structure_module.RVineMatrix.from_trees(
        4, selection.trees)
    assert not hasattr(selection, "fitted")
    assert isinstance(selection.trees, tuple)
    assert all(isinstance(level, tuple) for level in selection.trees)


def test_public_auto_replay_uses_canonical_order_and_preserves_fit_results():
    u = _gaussian_data(dimension=4)
    auto = VineCopula(
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    ).fit(u)
    canonical_trees = auto.structure.to_trees()
    specs = [
        [
            (
                type(auto.pair_copulas[
                    auto._matrix_key(tree, edge)].copula),
                auto.pair_copulas[
                    auto._matrix_key(tree, edge)].copula.rotate,
            )
            for edge, _ in enumerate(level)
        ]
        for tree, level in enumerate(canonical_trees)
    ]
    replay = VineCopula(
        structure=auto.structure,
        allow_rotations=False,
    ).fit(u, copulas=specs)

    assert auto.trees == canonical_trees
    assert replay.trees == canonical_trees
    assert replay.log_likelihood() == pytest.approx(
        auto.log_likelihood(), abs=1e-5)
    assert isinstance(replay.fit_result.nfev, int)
    assert isinstance(auto.fit_result.nfev, int)
    assert (
        replay.fit_result.actual_methods
        == auto.fit_result.actual_methods
    )
    for key, auto_pair in auto.pair_copulas.items():
        replay_pair = replay.pair_copulas[key]
        assert replay_pair.param == pytest.approx(auto_pair.param, abs=1e-6)
        assert isinstance(replay_pair.nfev, int)
        assert isinstance(auto_pair.nfev, int)
        assert (
            replay_pair.fit_diagnostics["actual_method"]
            == auto_pair.fit_diagnostics["actual_method"]
        )


@pytest.mark.parametrize(
    ("copula_class", "rotation"),
    (
        [
            (IndependentCopula, 0),
            (BivariateGaussianCopula, 0),
            (FrankCopula, 0),
        ]
        + [
            (copula_class, rotation)
            for copula_class in (GumbelCopula, ClaytonCopula, JoeCopula)
            for rotation in (0, 90, 180, 270)
        ]
    ),
)
def test_fixed_family_and_rotation_specs_are_preserved(
        copula_class, rotation):
    u = _gaussian_data(dimension=2)
    trees = dvine_structure(2).to_trees()

    fitted = fit_vine_edges(
        u,
        trees,
        copulas=[[(copula_class, rotation)]],
    )
    pair = fitted.pair_copulas[(0, 0)]

    assert type(pair.copula) is copula_class
    assert pair.copula.rotate == rotation
    assert pair.fit_result.method == "MLE"


def test_fixed_fit_truncation_and_threshold_use_independence_policy():
    u = _gaussian_data(dimension=4)
    trees = dvine_structure(4).to_trees()

    truncated = fit_vine_edges(
        u,
        trees,
        copulas=_fixed_specs(trees),
        truncation_level=1,
        truncation_fill="independent",
    )
    thresholded = fit_vine_edges(
        u,
        trees,
        copulas=_fixed_specs(trees),
        threshold=1.1,
    )

    assert all(
        isinstance(pair.copula, IndependentCopula)
        for (tree, _), pair in truncated.pair_copulas.items()
        if tree >= 1
    )
    assert all(
        isinstance(pair.copula, IndependentCopula)
        for pair in thresholded.pair_copulas.values()
    )


def test_fixed_structure_can_select_families_from_candidates():
    trees = dvine_structure(3).to_trees()

    fitted = fit_vine_edges(
        _gaussian_data(dimension=3),
        trees,
        candidates=[BivariateGaussianCopula],
        allow_rotations=False,
    )

    assert all(
        isinstance(
            pair.copula,
            (BivariateGaussianCopula, IndependentCopula),
        )
        for pair in fitted.pair_copulas.values()
    )
    assert any(
        isinstance(pair.copula, BivariateGaussianCopula)
        for pair in fitted.pair_copulas.values()
    )


def test_fixed_fit_min_loglik_and_truncated_mle_policies():
    u = _gaussian_data(rows=80, dimension=3)
    trees = dvine_structure(3).to_trees()
    specs = _fixed_specs(trees)

    pruned = fit_vine_edges(
        u,
        trees,
        copulas=specs,
        min_edge_logL=1e9,
    )
    truncated_mle = fit_vine_edges(
        u,
        trees,
        copulas=specs,
        method="gas",
        truncation_level=1,
        truncation_fill="mle",
    )

    assert all(
        isinstance(pair.copula, IndependentCopula)
        for pair in pruned.pair_copulas.values()
    )
    upper_pair = truncated_mle.pair_copulas[(1, 0)]
    assert upper_pair.fit_result.method == "MLE"
    assert upper_pair.fit_diagnostics["dynamic_attempted"] is False


def test_invalid_structure_is_rejected_before_edge_fit(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("optimizer/family selection must not run")

    monkeypatch.setattr(
        vine_fit_module, "_fit_with_strategy", unexpected_call)
    monkeypatch.setattr(
        vine_fit_module, "select_best_copula", unexpected_call)
    invalid_trees = [
        [
            (frozenset({0, 1}), frozenset()),
            (frozenset({1, 2}), frozenset()),
        ],
        [
            (frozenset({0, 2}), frozenset()),
        ],
    ]

    with pytest.raises(ValueError, match="conditioning variables"):
        fit_vine_edges(_gaussian_data(dimension=3), invalid_trees)


def test_vine_edge_fit_rejects_unknown_strategy_kwargs_before_selection(
        monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("edge selection/fitting must not run")

    monkeypatch.setattr(
        vine_fit_module, "_fit_with_strategy", unexpected_call)
    monkeypatch.setattr(
        vine_fit_module, "select_best_copula", unexpected_call)
    trees = dvine_structure(3).to_trees()

    with pytest.raises(
            TypeError,
            match="unexpected GAS keyword.*definitely_unknown"):
        fit_vine_edges(
            _gaussian_data(rows=20, dimension=3),
            trees,
            method="gas",
            fit_kwargs={"definitely_unknown": True},
        )


def test_vine_model_fit_rejects_unknown_strategy_kwargs_atomically():
    vine = VineCopula.dvine(d=3)

    with pytest.raises(
            TypeError,
            match="unexpected SCAR-TM-OU keyword.*definitely_unknown"):
        vine.fit(
            _gaussian_data(rows=20, dimension=3),
            method="scar-tm-ou",
            definitely_unknown=True,
        )

    assert vine.fit_result is None
    assert vine.pair_copulas is None
    assert getattr(vine, "_last_u", None) is None


def test_dynamic_vine_prefit_separates_constructor_and_initial_options(
        monkeypatch):
    calls = []
    original = vine_fit_module._fit_with_strategy

    def captured(copula, u_pair, method, config, fit_kwargs):
        calls.append((str(method).upper(), dict(fit_kwargs)))
        return original(copula, u_pair, method, config, fit_kwargs)

    monkeypatch.setattr(vine_fit_module, "_fit_with_strategy", captured)
    trees = dvine_structure(2).to_trees()
    fit_vine_edges(
        _gaussian_data(rows=35, dimension=2),
        trees,
        copulas=_fixed_specs(trees),
        method="scar-tm-ou",
        fit_kwargs={
            "K": 12,
            "alpha0": np.array([1.0, 0.0, 1.0]),
            "maxiter": 3,
        },
    )

    selection_kwargs = calls[0][1]
    dynamic_kwargs = calls[1][1]
    assert calls[0][0] == "MLE"
    assert "K" not in selection_kwargs
    assert "alpha0" not in selection_kwargs
    assert selection_kwargs["maxiter"] == 3
    assert calls[1][0] == "SCAR-TM-OU"
    assert dynamic_kwargs["K"] == 12
    np.testing.assert_array_equal(
        dynamic_kwargs["alpha0"],
        np.array([1.0, 0.0, 1.0]),
    )
    assert dynamic_kwargs["initial_mle_result"].method == "MLE"


def test_dynamic_failure_falls_back_to_mle_and_is_aggregated(monkeypatch):
    original = vine_fit_module._fit_with_strategy

    def failed_dynamic(copula, u_pair, method, config, fit_kwargs):
        if str(method).lower() == "gas":
            return GASResult(
                log_likelihood=-1e6,
                method="GAS",
                copula_name=copula.name,
                success=False,
                nfev=17,
                message="forced failure",
                params=gas_params(0.0, 0.0, 0.95),
            )
        return original(copula, u_pair, method, config, fit_kwargs)

    monkeypatch.setattr(
        vine_fit_module, "_fit_with_strategy", failed_dynamic)
    trees = dvine_structure(2).to_trees()
    fitted = fit_vine_edges(
        _gaussian_data(rows=100, dimension=2),
        trees,
        copulas=_fixed_specs(trees),
        method="gas",
    )

    pair = fitted.pair_copulas[(0, 0)]
    assert pair.fit_result.method == "MLE"
    assert pair.fit_diagnostics["dynamic_attempted"] is True
    assert pair.fit_diagnostics["attempted_method"] == "GAS"
    assert pair.fit_diagnostics["attempted_nfev"] == 17
    assert pair.fit_diagnostics["fallback_used"] is True
    assert fitted.actual_methods == {"MLE": 1}
    assert fitted.fallback_count == 1
    assert len(fitted.fallback_edges) == 1


def test_dynamic_failure_policy_keep_retains_unsuccessful_result(monkeypatch):
    original = vine_fit_module._fit_with_strategy

    def failed_dynamic(copula, u_pair, method, config, fit_kwargs):
        if str(method).lower() == "gas":
            return GASResult(
                log_likelihood=-123.0,
                method="GAS",
                copula_name=copula.name,
                success=False,
                nfev=17,
                message="forced failure",
                params=gas_params(0.0, 0.0, 0.95),
            )
        return original(copula, u_pair, method, config, fit_kwargs)

    monkeypatch.setattr(
        vine_fit_module, "_fit_with_strategy", failed_dynamic)
    trees = dvine_structure(2).to_trees()
    fitted = fit_vine_edges(
        _gaussian_data(rows=100, dimension=2),
        trees,
        copulas=_fixed_specs(trees),
        method="gas",
        dynamic_failure_policy="keep",
    )

    pair = fitted.pair_copulas[(0, 0)]
    assert pair.fit_result.method == "GAS"
    assert pair.fit_result.success is False
    assert pair.log_likelihood == -123.0
    assert pair.fit_diagnostics["fallback_used"] is False
    assert pair.fit_diagnostics["unsuccessful_dynamic_kept"] is True
    assert fitted.actual_methods == {"GAS": 1}
    assert fitted.fallback_count == 0


def test_invalid_dynamic_failure_policy_is_rejected():
    trees = dvine_structure(2).to_trees()
    with pytest.raises(ValueError, match="dynamic_failure_policy"):
        fit_vine_edges(
            _gaussian_data(rows=20, dimension=2),
            trees,
            dynamic_failure_policy="invalid",
        )


@pytest.mark.parametrize(
    ("method", "fit_kwargs"),
    [
        ("gas", {}),
        (
            "scar-tm-ou",
            {
                "K": 12,
                "grid_range": 3.0,
                "gtol": 5e-2,
                "maxiter": 20,
            },
        ),
    ],
)
def test_fixed_dynamic_methods_preserve_attempt_and_result_diagnostics(
        method, fit_kwargs):
    trees = dvine_structure(2).to_trees()
    fitted = fit_vine_edges(
        _gaussian_data(rows=45, dimension=2),
        trees,
        copulas=_fixed_specs(trees),
        method=method,
        fit_kwargs=fit_kwargs,
    )

    pair = fitted.pair_copulas[(0, 0)]
    assert pair.fit_diagnostics["requested_method"] == method.upper()
    assert pair.fit_diagnostics["dynamic_attempted"] is True
    assert pair.fit_diagnostics["attempted_method"] == method.upper()
    assert pair.fit_diagnostics["actual_method"] in {
        method.upper(),
        "MLE",
    }
    assert sum(fitted.actual_methods.values()) == 1


def test_vine_edge_fit_contains_canonical_state_and_aggregates():
    trees = cvine_structure(4).to_trees()
    fitted = fit_vine_edges(
        _gaussian_data(dimension=4),
        trees,
        copulas=_fixed_specs(trees),
    )

    assert isinstance(fitted, VineEdgeFit)
    assert sorted(fitted.pair_copulas) == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0),
    ]
    assert [
        len(level) for level in fitted.as_levels()
    ] == [3, 2, 1]
    assert fitted.log_likelihood == pytest.approx(sum(
        pair.log_likelihood for pair in fitted.pair_copulas.values()))
    assert fitted.parameter_count == sum(
        pair.n_params for pair in fitted.pair_copulas.values())
    assert fitted.fit_diagnostics["edge_count"] == 6
    assert sum(fitted.actual_methods.values()) == 6
