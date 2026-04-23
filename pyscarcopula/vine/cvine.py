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
from pyscarcopula.vine._edge import (
    VineEdge, _edge_h, _edge_log_likelihood, _get_alpha, _get_gas_params,
)
from pyscarcopula.vine._selection import (
    select_best_copula, _default_candidates,
)
from pyscarcopula.vine._helpers import (
    _clip_unit, generate_r_for_sample, generate_r_for_predict,
)
from pyscarcopula.vine._conditional_cvine import (
    ensure_cvine_conditional_supported,
    is_prefix_conditioning,
    sample_cvine_conditional_general_with_r,
    sample_cvine_conditional_prefix_with_r,
    validate_cvine_given,
)


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

    Estimation supports mixed models: strong edges (Tree 0-1) use
    SCAR-TM-OU for time-varying parameters, while weak edges (upper
    trees) fall back to MLE for efficiency.

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

    # ── Fit ────────────────────────────────────────────────────────

    def fit(self, data, method='mle', to_pobs=False,
            copulas=None, K=300, grid_range=5.0,
            truncation_level=None, min_edge_logL=None,
            transform_type='xtanh',
            **kwargs):
        """
        Fit the C-vine copula.

        Parameters
        ----------
        data : (T, d) array
        method : str — 'mle', 'gas', or SCAR method name
        to_pobs : bool
        copulas : None (auto-select) or list-of-lists of
                  (copula_class, rotation) tuples
        K : int — grid size for SCAR
        grid_range : float
        truncation_level : int or None
        min_edge_logL : float or None
        **kwargs : forwarded to copula.fit()

        Returns
        -------
        self
        """
        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        T, d = u.shape
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

                edge = VineEdge(tree=j, idx=i)

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
                    scar_kwargs = {kk: vv for kk, vv in kwargs.items()
                                   if kk not in ('alpha0', 'K', 'grid_range')}
                    result = _api_fit(cop, u_pair, method=method,
                                      K=K, grid_range=grid_range,
                                      alpha0=kwargs.get('alpha0'),
                                      **scar_kwargs)

                edge.copula = cop
                edge.fit_result = result
                self.edges[j].append(edge)

            if j < d - 2:
                for i in range(n_edges):
                    u1 = _clip_unit(v[j][0])
                    u2 = _clip_unit(v[j][i + 1])
                    u_pair = np.column_stack((u1, u2))
                    edge = self.edges[j][i]
                    v[j + 1][i] = _clip_unit(
                        _edge_h(edge, u2, u1, u_pair, K, grid_range))

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

    # ── Log-likelihood ────────────────────────────────────────────

    def log_likelihood(self, data, to_pobs=False, K=300, grid_range=5.0):
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
                total_ll += _edge_log_likelihood(edge, u_pair, K, grid_range)

            if j < d - 2:
                for i in range(n_edges):
                    u1 = _clip_unit(v[j][0])
                    u2 = _clip_unit(v[j][i + 1])
                    u_pair = np.column_stack((u1, u2))
                    edge = self.edges[j][i]
                    v[j + 1][i] = _clip_unit(
                        _edge_h(edge, u2, u1, u_pair, K, grid_range))

        return total_ll

    # ── Sampling ─────────────────────────────────────────────────

    def sample(self, n, u_train=None, K=300, grid_range=5.0, rng=None):
        """Sample from fitted C-vine using Rosenblatt inverse."""
        if self.edges is None:
            raise ValueError("Fit first")

        d = self.d
        if rng is None:
            rng = np.random.default_rng()

        from pyscarcopula._types import GASResult

        has_gas = any(
            isinstance(edge.fit_result, GASResult)
            for tree in self.edges for edge in tree)

        if has_gas:
            return self._sample_stepwise(n, d, rng)
        else:
            return self._sample_vectorized(n, d, rng)

    def _sample_vectorized(self, n, d, rng):
        """Vectorized sample for MLE/SCAR (no GAS edges)."""
        w = rng.uniform(0, 1, (n, d))
        x = np.zeros((n, d))
        v_samp = [[None] * d for _ in range(d)]

        x[:, 0] = w[:, 0]
        v_samp[0][0] = w[:, 0]

        r_sampling = [[None] * (d - 1) for _ in range(d - 1)]
        for j in range(d - 1):
            for i in range(d - j - 1):
                edge = self.edges[j][i]
                r_sampling[j][i] = generate_r_for_sample(edge, n, rng)

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
        """Step-by-step sample for GAS edges."""
        from pyscarcopula._types import GASResult, LatentResult, MLEResult
        from pyscarcopula.copula.independent import IndependentCopula

        x = np.zeros((n, d))

        n_trees = d - 1
        r_state = [[None] * (d - 1) for _ in range(n_trees)]
        f_state = [[None] * (d - 1) for _ in range(n_trees)]
        r_ou_path = [[None] * (d - 1) for _ in range(n_trees)]

        for j in range(n_trees):
            for i in range(d - j - 1):
                edge = self.edges[j][i]
                if isinstance(edge.copula, IndependentCopula):
                    r_state[j][i] = 0.0
                elif isinstance(edge.fit_result, MLEResult):
                    r_state[j][i] = edge.fit_result.copula_param
                elif isinstance(edge.fit_result, GASResult):
                    p = edge.fit_result.params
                    if abs(p.beta) < 1.0 - 1e-8:
                        f_state[j][i] = p.omega / (1.0 - p.beta)
                    else:
                        f_state[j][i] = p.omega
                    r_state[j][i] = float(
                        edge.copula.transform(
                            np.array([f_state[j][i]]))[0])
                elif isinstance(edge.fit_result, LatentResult):
                    r_ou_path[j][i] = generate_r_for_sample(edge, n, rng)
                    r_state[j][i] = float(r_ou_path[j][i][0])

        score_eps = 1e-4

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

            # Update GAS state
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

                    if isinstance(edge.fit_result, GASResult):
                        p = edge.fit_result.params
                        f_t = f_state[j][i]
                        r_t = r_state[j][i]
                        scaling = getattr(
                            edge.fit_result, 'scaling', 'unit')

                        u1a = np.array([u1_t])
                        u2a = np.array([u2_t])

                        f_plus = f_t + score_eps
                        f_minus = f_t - score_eps
                        r_plus = float(edge.copula.transform(
                            np.array([f_plus]))[0])
                        r_minus = float(edge.copula.transform(
                            np.array([f_minus]))[0])

                        ll_plus = float(edge.copula.log_pdf(
                            u1a, u2a, np.array([r_plus]))[0])
                        ll_minus = float(edge.copula.log_pdf(
                            u1a, u2a, np.array([r_minus]))[0])

                        nabla = (ll_plus - ll_minus) / (2.0 * score_eps)

                        if scaling == 'fisher':
                            ll_t = float(edge.copula.log_pdf(
                                u1a, u2a, np.array([r_t]))[0])
                            d2 = (ll_plus - 2.0 * ll_t + ll_minus
                                  ) / (score_eps ** 2)
                            fisher = max(-d2, 1e-6)
                            s_t = nabla / fisher
                        else:
                            s_t = nabla

                        s_t = np.clip(s_t, -100.0, 100.0)
                        f_new = p.omega + p.beta * f_t + p.alpha * s_t
                        f_new = np.clip(f_new, -50.0, 50.0)
                        f_state[j][i] = float(f_new)
                        r_state[j][i] = float(
                            edge.copula.transform(
                                np.array([f_new]))[0])

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
        """Alias for sample."""
        return self.sample(n, rng=rng)

    # ── Prediction ───────────────────────────────────────────────

    def predict(self, n, u=None, K=300, grid_range=5.0,
                given=None, horizon='next', rng=None,
                predictive_r_mode=None):
        """Conditional predict: sample from vine for next-step.

        `given` fixes selected variables in pseudo-observation space,
        e.g. ``given={2: 0.6}``. For GAS and SCAR-TM-OU edges, `horizon`
        selects the predictive state timing.
        """
        if self.edges is None:
            raise ValueError("Fit first")

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
        from pyscarcopula._types import GASResult, LatentResult
        needs_v_train = any(
            isinstance(self.edges[j][i].fit_result, (GASResult, LatentResult))
            for j in range(d - 1)
            for i in range(d - j - 1))

        if u_data is not None and needs_v_train:
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
                            _edge_h(edge, u2, u1, u_pair, K, grid_range))

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
                r_pred[j][i] = generate_r_for_predict(
                    edge, n, v_pair, K, grid_range, horizon=horizon,
                    rng=rng, predictive_r_mode=predictive_r_mode)

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

    # ── Summary ──────────────────────────────────────────────────

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
                if edge.method.upper() == 'MLE':
                    param = f"r={edge.fit_result.copula_param:.4f}"
                else:
                    alpha = _get_alpha(edge.fit_result)
                    param = f"alpha={alpha}"
                ll = edge.fit_result.log_likelihood
                total_ll += ll
                rot_str = f" rot={rot}" if rot != 0 else ""
                print(f"  Edge {edge.idx}: {name}{rot_str}, "
                      f"{param}, logL={ll:.2f}")
        print(f"\nTotal logL (sum of edges): {total_ll:.2f}")
