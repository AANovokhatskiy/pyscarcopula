"""
C-vine copula model.

Structure:
    Tree 0: pairs (1,2), (1,3), ..., (1,d)
    Tree j: pairs (j+1, i | 1..j) for i = j+2, ..., d

Each edge stores a fitted BivariateCopula (with rotation).

Usage:
    from pyscarcopula.vine.cvine import CVineCopula

    vine = CVineCopula()
    vine.fit(u, method='mle')

    vine.log_likelihood(u)
    samples = vine.sample(10000)
    predictions = vine.predict(10000)
"""

import numpy as np
from scipy.optimize import OptimizeResult

from pyscarcopula._utils import pobs
from pyscarcopula.vine._pair_copula import PairCopula
from pyscarcopula.vine._rvine_edges import (
    _edge_h,
    _edge_log_likelihood,
    _edge_r_for_predict,
    _edge_r_for_sample,
)
from pyscarcopula.vine._selection import (
    select_best_copula, _default_candidates,
)
from pyscarcopula.vine._helpers import (
    _clip_unit,
)
from pyscarcopula.vine._conditional_cvine import (
    ensure_cvine_conditional_supported,
    is_prefix_conditioning,
    sample_cvine_conditional_general_with_r,
    sample_cvine_conditional_prefix_with_r,
    validate_cvine_given,
)
from pyscarcopula.vine._edge_adapter import (
    edge_condition_sample_state,
    edge_copula,
    edge_has_dynamic_params,
    edge_model_sample_state,
    edge_param,
    edge_result,
    edge_state_param,
    result_param_items,
)


def _edge_param_from_result(result):
    """Return scalar fit parameter stored by point-parameter results."""
    value = getattr(result, 'copula_param', None)
    if value is None:
        return None
    return float(value)


class CVineCopula:
    """
    C-vine copula for d dimensions.

    Decomposes d-dimensional dependence into d(d-1)/2 bivariate copulas
    arranged in a tree structure. Each edge copula can be from a different
    family (Gumbel, Clayton, Frank, Joe, Gaussian, or Independent),
    selected automatically via AIC/BIC.

    Tree structure (0-indexed):
        Tree j, edge i: copula for pair (j+1, j+i+2 | 1..j+1)
        Tree 0: unconditional pairs (1,2), (1,3), ..., (1,d)
        Tree 1: conditional pairs (2,3|1), (2,4|1), ...
        etc.

    Estimation supports mixed fitted strategies on different edges.

    Parameters
    ----------
    candidates : list of copula classes, or None (default: 5 families)
    allow_rotations : bool (default True)
    criterion : 'aic', 'bic', or 'loglik'
    """

    def __init__(self, candidates=None, allow_rotations=True,
                 criterion='aic'):
        self.candidates = candidates
        self.allow_rotations = allow_rotations
        self.criterion = criterion
        self.edges = None
        self.d = None
        self.method = None

    def _get_candidates(self):
        if self.candidates is not None:
            return self.candidates
        return _default_candidates()

    # Fit

    def fit(self, data, method='mle', to_pobs=False,
            copulas=None,
            truncation_level=None, min_edge_logL=None,
            transform_type='softplus',
            **kwargs):
        """
        Fit the C-vine copula.

        Parameters
        ----------
        data : (T, d) array
        method : str
            Strategy name forwarded to bivariate copula fitting.
        to_pobs : bool
        copulas : None (auto-select) or list-of-lists of
                  (copula_class, rotation) tuples
        truncation_level : int or None
        min_edge_logL : float or None
        **kwargs : forwarded to pair-copula strategy fit()

        Returns
        -------
        self
        """
        u = np.asarray(data, dtype=np.float64)
        if u.ndim != 2:
            raise ValueError(f"CVineCopula.fit: data must be 2D, got shape {u.shape}")
        if to_pobs:
            u = pobs(u)

        T, d = u.shape
        if d < 2:
            raise ValueError(f"CVineCopula.fit: need d >= 2, got d={d}")
        self.d = d
        self.method = method.upper()

        v = [[None] * d for _ in range(d)]
        for i in range(d):
            v[0][i] = u[:, i].copy()

        self.edges = [[] for _ in range(d - 1)]

        for j in range(d - 1):
            n_edges = d - j - 1

            for i in range(n_edges):
                u1 = _clip_unit(v[j][0])
                u2 = _clip_unit(v[j][i + 1])
                u_pair = np.column_stack((u1, u2))

                # Step 1: select copula family (always via MLE)
                if copulas is not None:
                    cop_class, rotation = copulas[j][i]
                    try:
                        cop = cop_class(rotate=rotation,
                                        transform_type=transform_type)
                    except TypeError:
                        cop = cop_class(rotate=rotation)
                    from pyscarcopula.copula.independent import IndependentCopula
                    if isinstance(cop, IndependentCopula):
                        result = cop.fit(u_pair)
                    else:
                        from pyscarcopula.api import fit as _api_fit
                        result = _api_fit(cop, u_pair, method='mle')
                else:
                    cop, result = select_best_copula(
                        u1, u2, self._get_candidates(),
                        self.allow_rotations, self.criterion,
                        transform_type=transform_type)
                selection_result = result

                # Step 2: decide whether to refit with dynamic method
                from pyscarcopula.copula.independent import IndependentCopula
                skip_dynamic = (
                    self.method == 'MLE'
                    or isinstance(cop, IndependentCopula)
                    or (truncation_level is not None
                        and j >= truncation_level)
                    or (min_edge_logL is not None
                        and result.log_likelihood < min_edge_logL)
                )

                if not skip_dynamic:
                    from pyscarcopula.api import fit as _api_fit
                    dynamic_result = _api_fit(
                        cop, u_pair, method=method, **kwargs)
                    if bool(getattr(dynamic_result, 'success', True)):
                        result = dynamic_result
                    else:
                        result = selection_result

                edge = PairCopula(
                    copula=cop,
                    param=_edge_param_from_result(result),
                    log_likelihood=float(result.log_likelihood),
                    nfev=int(getattr(result, 'nfev', 0)),
                    tau=0.0,
                    fit_result=result,
                    tree=j,
                    idx=i,
                )
                self.edges[j].append(edge)

            if j < d - 2:
                for i in range(n_edges):
                    u1 = _clip_unit(v[j][0])
                    u2 = _clip_unit(v[j][i + 1])
                    u_pair = np.column_stack((u1, u2))
                    edge = self.edges[j][i]
                    v[j + 1][i] = _clip_unit(
                        _edge_h(edge, u2, u1, u_pair=u_pair))

        total_ll = sum(e.fit_result.log_likelihood
                       for tree in self.edges for e in tree)
        total_nfev = sum(getattr(e.fit_result, 'nfev', 0)
                         for tree in self.edges for e in tree)
        n_edges_total = sum(len(tree) for tree in self.edges)

        self.fit_result = OptimizeResult()
        self.fit_result.log_likelihood = total_ll
        self.fit_result.method = method
        self.fit_result.name = f"C-vine ({d}d, {n_edges_total} edges)"
        self.fit_result.nfev = total_nfev
        self.fit_result.success = True
        self._last_u = u

        return self

    # Log-likelihood

    def log_likelihood(self, data, to_pobs=False, **kwargs):
        """Compute total log-likelihood of the C-vine."""
        if self.edges is None:
            raise ValueError("Fit first")

        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        T, d = u.shape
        v = [[None] * d for _ in range(d)]
        for i in range(d):
            v[0][i] = u[:, i].copy()

        total_ll = 0.0

        for j in range(d - 1):
            n_edges = d - j - 1

            for i in range(n_edges):
                u1 = _clip_unit(v[j][0])
                u2 = _clip_unit(v[j][i + 1])
                u_pair = np.column_stack((u1, u2))
                edge = self.edges[j][i]
                total_ll += _edge_log_likelihood(edge, u_pair, **kwargs)

            if j < d - 2:
                for i in range(n_edges):
                    u1 = _clip_unit(v[j][0])
                    u2 = _clip_unit(v[j][i + 1])
                    u_pair = np.column_stack((u1, u2))
                    edge = self.edges[j][i]
                    v[j + 1][i] = _clip_unit(
                        _edge_h(edge, u2, u1, u_pair=u_pair, **kwargs))

        return total_ll

    # Sampling

    def sample(self, n, u_train=None, rng=None, **kwargs):
        """Sample from fitted C-vine using Rosenblatt inverse.

        ``u_train`` and extra keyword arguments are accepted for legacy API
        compatibility and are not used by the model-reproduction sampler.
        """
        if self.edges is None:
            raise ValueError("Fit first")
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(f"CVineCopula.sample: n must be positive int, got {n!r}")

        d = self.d
        n = int(n)
        if rng is None:
            rng = np.random.default_rng()

        has_stateful = any(
            edge_model_sample_state(edge.copula, edge.fit_result) is not None
            for tree in self.edges for edge in tree)

        if has_stateful:
            return self._sample_stepwise(n, d, rng)
        else:
            return self._sample_vectorized(n, d, rng)

    def _sample_vectorized(self, n, d, rng):
        """Vectorized sample for edges without strategy-owned state."""
        w = rng.uniform(0, 1, (n, d))
        x = np.zeros((n, d))
        v_samp = [[None] * d for _ in range(d)]

        x[:, 0] = w[:, 0]
        v_samp[0][0] = w[:, 0]

        r_sampling = [[None] * (d - 1) for _ in range(d - 1)]
        for j in range(d - 1):
            for i in range(d - j - 1):
                edge = self.edges[j][i]
                r_sampling[j][i] = _edge_r_for_sample(edge, n, rng)

        for i in range(1, d):
            v_samp[i][0] = w[:, i]

            for k in range(i - 1, -1, -1):
                edge = self.edges[k][i - k - 1]
                r = r_sampling[k][i - k - 1]
                v_samp[i][0] = _clip_unit(
                    edge.copula.h_inverse(v_samp[i][0], v_samp[k][k], r))

            x[:, i] = v_samp[i][0]

            if i < d - 1:
                for j_idx in range(i):
                    edge = self.edges[j_idx][i - j_idx - 1]
                    r = r_sampling[j_idx][i - j_idx - 1]
                    v_samp[i][j_idx + 1] = _clip_unit(
                        edge.copula.h(v_samp[i][j_idx],
                                      v_samp[j_idx][j_idx], r))

        return x

    def _sample_stepwise(self, n, d, rng):
        """Step-by-step sample for edges with strategy-owned state."""
        from pyscarcopula.copula.independent import IndependentCopula

        x = np.zeros((n, d))

        n_trees = d - 1
        r_state = [[None] * (d - 1) for _ in range(n_trees)]
        model_state = [[None] * (d - 1) for _ in range(n_trees)]
        r_ou_path = [[None] * (d - 1) for _ in range(n_trees)]

        for j in range(n_trees):
            for i in range(d - j - 1):
                edge = self.edges[j][i]
                copula = edge_copula(edge)
                result = edge_result(edge)
                state = edge_model_sample_state(copula, result)
                if isinstance(copula, IndependentCopula):
                    r_state[j][i] = 0.0
                elif state is not None:
                    model_state[j][i] = state
                    r_state[j][i] = edge_state_param(state)
                else:
                    r_ou_path[j][i] = _edge_r_for_sample(edge, n, rng)
                    r_state[j][i] = float(r_ou_path[j][i][0])

        for t in range(n):
            for j in range(n_trees):
                for i in range(d - j - 1):
                    if r_ou_path[j][i] is not None:
                        r_state[j][i] = float(r_ou_path[j][i][t])

            w = rng.uniform(0, 1, d)
            v_samp = [[None] * d for _ in range(d)]

            x[t, 0] = w[0]
            v_samp[0][0] = w[0]

            for i in range(1, d):
                val = w[i]

                for k in range(i - 1, -1, -1):
                    edge = self.edges[k][i - k - 1]
                    r_k = r_state[k][i - k - 1]
                    val = float(_clip_unit(np.atleast_1d(
                        edge.copula.h_inverse(
                            np.array([val]),
                            np.array([v_samp[k][k]]),
                            np.array([r_k]))))[0])

                x[t, i] = val
                v_samp[i][0] = val

                if i < d - 1:
                    for j_idx in range(i):
                        edge = self.edges[j_idx][i - j_idx - 1]
                        r_ji = r_state[j_idx][i - j_idx - 1]
                        v_samp[i][j_idx + 1] = float(_clip_unit(
                            np.atleast_1d(
                                edge.copula.h(
                                    np.array([v_samp[i][j_idx]]),
                                    np.array([v_samp[j_idx][j_idx]]),
                                    np.array([r_ji]))))[0])

            # Update strategy-owned state.
            v_obs = [[None] * d for _ in range(d)]
            for i in range(d):
                v_obs[0][i] = x[t, i]

            for j in range(n_trees):
                for i in range(d - j - 1):
                    edge = self.edges[j][i]
                    u1_t = float(_clip_unit(
                        np.atleast_1d(v_obs[j][0]))[0])
                    u2_t = float(_clip_unit(
                        np.atleast_1d(v_obs[j][i + 1]))[0])

                    if model_state[j][i] is not None:
                        u1a = np.array([u1_t])
                        u2a = np.array([u2_t])
                        model_state[j][i] = edge_condition_sample_state(
                            edge.copula,
                            edge.fit_result,
                            model_state[j][i],
                            np.column_stack((u1a, u2a)),
                        )
                        r_state[j][i] = edge_state_param(model_state[j][i])

                    if j < n_trees - 1:
                        r_ji = r_state[j][i]
                        h_val = float(_clip_unit(np.atleast_1d(
                            edge.copula.h(
                                np.array([u2_t]),
                                np.array([u1_t]),
                                np.array([r_ji]))
                        ))[0])
                        v_obs[j + 1][i] = h_val

        return x

    def sample_model(self, n, u=None, rng=None):
        """Legacy alias for sample.

        ``u`` is accepted for backward compatibility and ignored.
        """
        return self.sample(n, rng=rng)

    # Prediction

    def save(self, path, *, include_data=True):
        """Save this fitted C-vine model to disk."""
        from pyscarcopula.io import save_model

        save_model(self, path, include_data=include_data)

    @classmethod
    def load(cls, path):
        """Load a saved C-vine model from disk."""
        from pyscarcopula.io import load_model

        return load_model(path, expected_type=cls)

    def predict(self, n, u=None,
                given=None, horizon='next', rng=None,
                predictive_r_mode=None, **kwargs):
        """Conditional predict: sample from vine for next-step.

        `given` fixes selected variables in pseudo-observation space,
        e.g. ``given={2: 0.6}``. For dynamic edges, `horizon` selects the
        predictive state timing.
        """
        if self.edges is None:
            raise ValueError("Fit first")
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(f"CVineCopula.predict: n must be positive int, got {n!r}")
        n = int(n)

        u_data = u if u is not None else getattr(self, '_last_u', None)
        given = validate_cvine_given(given, self.d)

        d = self.d
        if rng is None:
            rng = np.random.default_rng()

        if len(given) == d:
            out = np.empty((n, d), dtype=np.float64)
            for i in range(d):
                out[:, i] = given[i]
            return out

        # Build v_train if needed for dynamic predictive edge states.
        v_train = None
        needs_v_train = any(
            edge_has_dynamic_params(self.edges[j][i])
            for j in range(d - 1)
            for i in range(d - j - 1))

        if u_data is not None and needs_v_train:
            u_data = np.asarray(u_data, dtype=np.float64)
            if u_data.ndim != 2 or u_data.shape[1] != d:
                raise ValueError(
                    f"CVineCopula.predict: u must be (T, {d}), "
                    f"got {u_data.shape}")
            v_train = [[None] * d for _ in range(d)]
            for ii in range(d):
                v_train[0][ii] = u_data[:, ii].copy()
            for j in range(d - 1):
                for ii in range(d - j - 1):
                    u1 = _clip_unit(v_train[j][0])
                    u2 = _clip_unit(v_train[j][ii + 1])
                    u_pair = np.column_stack((u1, u2))
                    edge = self.edges[j][ii]
                    if j < d - 2:
                        v_train[j + 1][ii] = _clip_unit(
                            _edge_h(edge, u2, u1, u_pair=u_pair, **kwargs))

        # Precompute r for each edge
        r_pred = [[None] * (d - 1) for _ in range(d - 1)]
        for j in range(d - 1):
            for i in range(d - j - 1):
                edge = self.edges[j][i]
                v_pair = None
                if v_train is not None:
                    u1 = _clip_unit(v_train[j][0])
                    u2 = _clip_unit(v_train[j][i + 1])
                    v_pair = np.column_stack((u1, u2))
                r_pred[j][i] = _edge_r_for_predict(
                    edge,
                    n,
                    u_train_pair=v_pair,
                    horizon=horizon,
                    rng=rng,
                    predictive_r_mode=predictive_r_mode,
                )

        if given:
            ensure_cvine_conditional_supported(self)
            if is_prefix_conditioning(given):
                return sample_cvine_conditional_prefix_with_r(
                    self, n, r_pred, given, rng)
            return sample_cvine_conditional_general_with_r(
                self, n, r_pred, given, rng)

        w = rng.uniform(0, 1, (n, d))
        x = np.zeros((n, d))
        v_samp = [[None] * d for _ in range(d)]

        x[:, 0] = w[:, 0]
        v_samp[0][0] = w[:, 0]

        for i in range(1, d):
            v_samp[i][0] = w[:, i]

            for k in range(i - 1, -1, -1):
                edge = self.edges[k][i - k - 1]
                r = r_pred[k][i - k - 1]
                v_samp[i][0] = _clip_unit(
                    edge.copula.h_inverse(v_samp[i][0], v_samp[k][k], r))

            x[:, i] = v_samp[i][0]

            if i < d - 1:
                for j_idx in range(i):
                    edge = self.edges[j_idx][i - j_idx - 1]
                    r = r_pred[j_idx][i - j_idx - 1]
                    v_samp[i][j_idx + 1] = _clip_unit(
                        edge.copula.h(v_samp[i][j_idx],
                                      v_samp[j_idx][j_idx], r))

        return x

    # Summary

    def summary(self):
        """Print vine structure summary."""
        if self.edges is None:
            print("Not fitted")
            return

        print(f"C-Vine Copula (d={self.d}, method={self.method})")
        print("=" * 60)
        total_ll = 0.0
        for j, tree_edges in enumerate(self.edges):
            print(f"\nTree {j}:")
            for edge in tree_edges:
                cop = edge.copula
                name = cop.name
                rot = cop._rotate
                items = result_param_items(edge_result(edge))
                if items:
                    param = ", ".join(
                        f"{name}={float(value):.4f}"
                        for name, value in items)
                else:
                    param = f"r={edge_param(edge, default=0.0):.4f}"
                ll = edge.fit_result.log_likelihood
                total_ll += ll
                rot_str = f" rot={rot}" if rot != 0 else ""
                print(f"  Edge {edge.idx}: {name}{rot_str}, "
                      f"{param}, logL={ll:.2f}")
        print(f"\nTotal logL (sum of edges): {total_ll:.2f}")
