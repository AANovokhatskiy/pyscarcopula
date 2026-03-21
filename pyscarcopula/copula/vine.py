"""
C-vine copula model.

Structure:
    Tree 0: pairs (1,2), (1,3), ..., (1,d)
    Tree j: pairs (j+1, i | 1..j) for i = j+2, ..., d

Each edge stores a fitted BivariateCopula (with rotation).

Usage:
    from pyscarcopula.vine import CVineCopula
    from pyscarcopula import GumbelCopula, ClaytonCopula, FrankCopula

    vine = CVineCopula()
    vine.fit(u, method='mle')

    vine.log_likelihood(u)
    samples = vine.sample(10000)
    predictions = vine.predict(10000)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from pyscarcopula.utils import pobs


# ══════════════════════════════════════════════════════════════════
# Edge: stores one fitted bivariate copula
# ══════════════════════════════════════════════════════════════════

@dataclass
class VineEdge:
    """One edge in the vine: a fitted bivariate copula."""
    tree: int             # tree level (0-indexed)
    idx: int              # edge index within tree
    copula: object = None # fitted BivariateCopula instance
    fit_result: object = None
    copula_param: object = None  # scalar (MLE) or array (SCAR alpha)

    @property
    def method(self):
        if self.fit_result is None:
            return None
        return self.fit_result.method

    def get_r(self, u_pair, T=None):
        """
        Get copula parameter r for given data.
        MLE: constant.
        SCAR: smoothed_params via TM forward pass.
        """
        method = self.method
        if method is None:
            raise ValueError("Edge not fitted")

        if method.upper() == 'MLE':
            n = T if T is not None else len(u_pair)
            return np.full(n, self.fit_result.copula_param)
        else:
            return self.copula.smoothed_params(u_pair)

    def get_r_predict(self, n):
        """
        Get copula parameter r for prediction (sampling from x_T).
        MLE: constant.
        SCAR: sample from stationary OU.
        """
        method = self.method
        if method.upper() == 'MLE':
            return np.full(n, self.fit_result.copula_param)
        else:
            alpha = self.fit_result.alpha
            theta, mu, nu = alpha
            sigma2 = nu ** 2 / (2.0 * theta)
            x_T = np.random.normal(mu, np.sqrt(sigma2), n)
            return self.copula.transform(x_T)


# ══════════════════════════════════════════════════════════════════
# Default copula candidates
# ══════════════════════════════════════════════════════════════════

def _default_candidates():
    """Default set of bivariate copula classes to try."""
    from pyscarcopula import (GumbelCopula, ClaytonCopula, FrankCopula,
                                JoeCopula, BivariateGaussianCopula)
    return [GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
            BivariateGaussianCopula]

def _all_rotations(copula_class):
    """Get valid rotations for a copula class."""
    cop = copula_class()
    if hasattr(cop, 'rotatable') and not cop.rotatable:
        return [0]
    try:
        copula_class(rotate=180)
        return [0, 90, 180, 270]
    except (ValueError, TypeError):
        return [0]


# ══════════════════════════════════════════════════════════════════
# Copula selection for one edge
# ══════════════════════════════════════════════════════════════════

def select_best_copula(u1, u2, candidates, allow_rotations=True,
                       criterion='aic'):
    """
    Select best bivariate copula for (u1, u2) by AIC/BIC/logL.

    Always includes IndependentCopula as a baseline competitor.
    If no parametric copula beats independence, the edge is
    set to independent (zero cost for SCAR/GAS).

    Parameters
    ----------
    u1, u2 : (T,) arrays
    candidates : list of copula classes
    allow_rotations : bool
    criterion : 'aic', 'bic', or 'loglik'

    Returns
    -------
    best_copula : fitted BivariateCopula instance
    best_result : fit result
    """
    from pyscarcopula.copula.independent import IndependentCopula

    u_pair = np.column_stack((u1, u2))
    T = len(u1)

    # Start with independence as baseline (AIC=0, BIC=0, logL=0)
    indep = IndependentCopula()
    indep_result = indep.fit(u_pair)

    if criterion == 'aic':
        best_score = 0.0   # AIC = -2*0 + 2*0 = 0
    elif criterion == 'bic':
        best_score = 0.0   # BIC = -2*0 + 0*log(T) = 0
    else:
        best_score = 0.0   # -logL = 0

    best_copula = indep
    best_result = indep_result

    for cop_class in candidates:
        # Skip IndependentCopula if it's in candidates (already baseline)
        if cop_class is IndependentCopula:
            continue

        rotations = _all_rotations(cop_class) if allow_rotations else [0]

        for angle in rotations:
            try:
                cop = cop_class(rotate=angle)
                result = cop.fit(u_pair, method='mle')
                logL = result.log_likelihood
                n_params = 1  # MLE has 1 parameter for archimedean

                if criterion == 'aic':
                    score = -2 * logL + 2 * n_params
                elif criterion == 'bic':
                    score = -2 * logL + n_params * np.log(T)
                else:  # loglik
                    score = -logL

                if score < best_score:
                    best_score = score
                    best_copula = cop
                    best_result = result
            except Exception:
                continue

    return best_copula, best_result


# ══════════════════════════════════════════════════════════════════
# Helper: h-function dispatch (MLE / GAS / SCAR)
# ══════════════════════════════════════════════════════════════════

def _edge_h(edge, u2, u1, u_pair, K=300, grid_range=5.0):
    """
    Compute h(u2 | u1; r) for a vine edge using the correct method.

    MLE:  h(u2, u1; theta_mle) — constant parameter.
    GAS:  h(u2, u1; Psi(f_t)) — along deterministic GAS path.
    SCAR: E[h(u2, u1; Psi(x)) | data] — mixture over predictive
          distribution via transfer matrix forward pass.
    Independent: h(u2 | u1) = u2 — trivial pass-through.
    """
    from pyscarcopula.copula.independent import IndependentCopula
    if isinstance(edge.copula, IndependentCopula):
        return u2.copy()

    method = edge.method.upper() if edge.method else 'MLE'

    if method == 'MLE':
        r = edge.get_r(u_pair)
        return edge.copula.h(u2, u1, r)

    elif method == 'GAS':
        from pyscarcopula.latent.gas_process import _gas_mixture_h
        alpha = edge.fit_result.alpha
        scaling = getattr(edge.fit_result, 'scaling', 'unit')
        return _gas_mixture_h(alpha[0], alpha[1], alpha[2],
                              u_pair, edge.copula, scaling)

    else:
        # SCAR-TM-OU: mixture h via transfer matrix
        from pyscarcopula.latent.ou_process import _tm_forward_mixture_h
        alpha = edge.fit_result.alpha
        theta, mu, nu = alpha
        return _tm_forward_mixture_h(theta, mu, nu, u_pair,
                                      edge.copula, K, grid_range)


# ══════════════════════════════════════════════════════════════════
# Helper: edge log-likelihood dispatch
# ══════════════════════════════════════════════════════════════════

def _edge_log_likelihood(edge, u_pair):
    """
    Compute log-likelihood for one edge using the correct method.

    MLE:  sum log c(u1, u2; theta_mle)
    GAS:  sum log c(u1, u2; Psi(f_t))  (score-driven filter)
    SCAR: log integral (transfer matrix likelihood)
    """
    method = edge.method.lower()
    cop = edge.copula
    alpha = edge.fit_result.alpha if hasattr(edge.fit_result, 'alpha') else edge.fit_result.copula_param

    if method == 'mle':
        alpha = edge.fit_result.copula_param
    
    ll = cop.mlog_likelihood(alpha=alpha, u=u_pair, method=method)
    return -ll  # mlog_likelihood returns minus log-likelihood


# ══════════════════════════════════════════════════════════════════
# C-Vine Copula
# ══════════════════════════════════════════════════════════════════

def _clip_unit(x):
    """Clip to (eps, 1-eps) to avoid NaN in h-functions."""
    eps = 1e-10
    return np.clip(x, eps, 1.0 - eps)


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
    trees) fall back to MLE for efficiency. The GoF test handles
    this correctly via per-edge h-function dispatch.

    Key parameters for vine.fit():
        method : 'mle', 'scar-tm-ou', 'gas'
        truncation_level : int — trees >= level stay MLE
        min_edge_logL : float — edges with MLE logL < threshold stay MLE
        tol : float — L-BFGS-B tolerance (5e-2 recommended for vine)

    Example
    -------
    >>> vine = CVineCopula(criterion='aic')
    >>> vine.fit(u, method='scar-tm-ou', min_edge_logL=10)
    >>> vine.summary()
    >>> gof = vine_gof_test(vine, u)

    Parameters
    ----------
    candidates : list of copula classes, or None (default: 5 families)
    allow_rotations : bool (default True)
    criterion : 'aic', 'bic', or 'loglik'
    """

    def __init__(self, candidates=None, allow_rotations=True,
                 criterion='aic'):
        """
        Parameters
        ----------
        candidates : list of BivariateCopula classes, or None (default 4)
        allow_rotations : bool
        criterion : 'aic', 'bic', 'loglik' — for automatic selection
        """
        self.candidates = candidates
        self.allow_rotations = allow_rotations
        self.criterion = criterion
        self.edges = None  # list of lists: edges[j][i] = VineEdge
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
            **kwargs):
        """
        Fit the C-vine copula.

        Parameters
        ----------
        data : (T, d) array
        method : str — 'mle', 'gas', or SCAR method name
        to_pobs : bool
        copulas : None (auto-select) or list-of-lists of
                  (copula_class, rotation) tuples for manual specification.
                  copulas[j][i] = (GumbelCopula, 180)
        K : int — grid size for SCAR transfer matrix (ignored for MLE/GAS)
        grid_range : float — grid range for SCAR (ignored for MLE/GAS)
        truncation_level : int or None
            If set, edges at tree levels >= truncation_level keep their
            MLE fit and are not refitted with SCAR/GAS. This is the
            standard vine truncation approach (Joe 2014). Example:
            truncation_level=2 means only trees 0 and 1 use SCAR.
        min_edge_logL : float or None
            If set, edges whose MLE log-likelihood is below this
            threshold are not refitted with SCAR/GAS. Weak edges
            contribute little to the total logL but cost the same
            to fit. Typical value: 5-10.
        **kwargs : forwarded to copula.fit() for SCAR/GAS methods

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

        # Initialize pseudo-obs matrix:
        # v[j][i] = (T,) array, i-th variable at tree level j
        # v[0][i] = u[:, i]  for i = 0..d-1
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
                    # Manual specification
                    cop_class, rotation = copulas[j][i]
                    cop = cop_class(rotate=rotation)
                    result = cop.fit(u_pair, method='mle')
                else:
                    # Automatic selection via AIC/BIC
                    cop, result = select_best_copula(
                        u1, u2, self._get_candidates(),
                        self.allow_rotations, self.criterion)

                # Step 2: decide whether to refit with dynamic method
                from pyscarcopula.copula.independent import IndependentCopula
                skip_dynamic = (
                    self.method == 'MLE'
                    or isinstance(cop, IndependentCopula)
                    or (truncation_level is not None and j >= truncation_level)
                    or (min_edge_logL is not None
                        and result.log_likelihood < min_edge_logL)
                )

                if not skip_dynamic:
                    scar_kwargs = {kk: vv for kk, vv in kwargs.items()
                                   if kk != 'alpha0'}
                    result = cop.fit(u_pair, method=method,
                                     alpha0=kwargs.get('alpha0'),
                                     **scar_kwargs)

                edge.copula = cop
                edge.fit_result = result
                edge.copula_param = (result.alpha
                                     if hasattr(result, 'alpha')
                                     and len(np.atleast_1d(result.alpha)) == 3
                                     else result.copula_param)

                self.edges[j].append(edge)

            # Compute pseudo-obs for next tree level using correct
            # h-function for each method:
            #   MLE:  h(u2, u1; theta_mle)
            #   GAS:  h along GAS-filtered path
            #   SCAR: mixture h (integral over predictive distribution)
            if j < d - 2:
                for i in range(n_edges):
                    u1 = _clip_unit(v[j][0])
                    u2 = _clip_unit(v[j][i + 1])
                    u_pair = np.column_stack((u1, u2))

                    edge = self.edges[j][i]
                    v[j + 1][i] = _clip_unit(
                        _edge_h(edge, u2, u1, u_pair, K, grid_range))

        return self

    # ── Log-likelihood ────────────────────────────────────────────

    def log_likelihood(self, data, to_pobs=False, K=300, grid_range=5.0):
        """
        Compute total log-likelihood of the C-vine.

        For each edge, uses the correct likelihood:
          MLE:  sum log c(u1, u2; theta)
          GAS:  sum log c(u1, u2; Psi(f_t))
          SCAR: transfer matrix integrated likelihood

        Pseudo-observations for higher trees are computed using
        the correct h-function (mixture h for SCAR, GAS-filtered
        h for GAS, constant-parameter h for MLE).
        """
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
                total_ll += _edge_log_likelihood(edge, u_pair)

            # Pseudo-obs for next tree: same h-dispatch as in fit()
            if j < d - 2:
                for i in range(n_edges):
                    u1 = _clip_unit(v[j][0])
                    u2 = _clip_unit(v[j][i + 1])
                    u_pair = np.column_stack((u1, u2))

                    edge = self.edges[j][i]
                    v[j + 1][i] = _clip_unit(
                        _edge_h(edge, u2, u1, u_pair, K, grid_range))

        return total_ll

    # ── Sampling (using fitted parameters on training data) ───────

    def sample(self, n, u_train=None, K=300, grid_range=5.0):
        """
        Sample from fitted C-vine using Rosenblatt inverse.

        For SCAR: uses mixture_h from u_train (must provide).
        For GAS:  uses GAS-filtered h from u_train.
        For MLE:  u_train not needed.

        Parameters
        ----------
        n : int — number of samples
        u_train : (T, d) or None — training data for dynamic models
        K : int — grid size for SCAR mixture h
        grid_range : float

        Returns
        -------
        x : (n, d) — samples in [0,1]^d
        """
        if self.edges is None:
            raise ValueError("Fit first")

        d = self.d
        w = np.random.uniform(0, 1, (n, d))
        x = np.zeros((n, d))
        # v_samp[j][i] = (n,) — intermediate h-transformed values
        v_samp = [[None] * d for _ in range(d)]

        x[:, 0] = w[:, 0]
        v_samp[0][0] = w[:, 0]

        # Precompute r for all edges (from training data)
        # For MLE: constant r
        # For GAS/SCAR: we need the h-transformed pseudo-obs from
        # training data to get the last parameter value
        r_cache = [[None] * (d - 1) for _ in range(d - 1)]
        if self.method != 'MLE' and u_train is not None:
            v_train = [[None] * d for _ in range(d)]
            for i in range(d):
                v_train[0][i] = u_train[:, i].copy()

            for j in range(d - 1):
                for i in range(d - j - 1):
                    u1 = _clip_unit(v_train[j][0])
                    u2 = _clip_unit(v_train[j][i + 1])
                    u_pair = np.column_stack((u1, u2))
                    edge = self.edges[j][i]
                    r_cache[j][i] = edge.get_r(u_pair)

                    if j < d - 2:
                        # Use correct h-function for pseudo-obs
                        v_train[j + 1][i] = _clip_unit(
                            _edge_h(edge, u2, u1, u_pair, K, grid_range))

        def _get_r_for_sampling(j, i, n_samples):
            """Get scalar r for sampling (last value of smoothed, or constant)."""
            edge = self.edges[j][i]
            if self.method == 'MLE':
                return np.full(n_samples, edge.fit_result.copula_param)
            elif r_cache[j][i] is not None:
                # Use last smoothed value for all samples
                return np.full(n_samples, r_cache[j][i][-1])
            else:
                return edge.get_r_predict(n_samples)

        for i in range(1, d):
            v_samp[i][0] = w[:, i]

            for k in range(i - 1, -1, -1):
                edge = self.edges[k][i - k - 1]
                r = _get_r_for_sampling(k, i - k - 1, n)
                v_samp[i][0] = _clip_unit(
                    edge.copula.h_inverse(v_samp[i][0], v_samp[k][k], r))

            x[:, i] = v_samp[i][0]

            if i < d - 1:
                for j_idx in range(i):
                    edge = self.edges[j_idx][i - j_idx - 1]
                    r = _get_r_for_sampling(j_idx, i - j_idx - 1, n)
                    v_samp[i][j_idx + 1] = _clip_unit(
                        edge.copula.h(v_samp[i][j_idx], v_samp[j_idx][j_idx], r))

        return x

    # ── Prediction (sample from x_T distribution) ────────────────

    def predict(self, n):
        """
        Predict: sample using x_T from latent process.

        For MLE: same as sample (constant params).
        For SCAR: samples x_T from stationary OU distribution.

        Parameters
        ----------
        n : int

        Returns
        -------
        x : (n, d)
        """
        if self.edges is None:
            raise ValueError("Fit first")

        d = self.d
        w = np.random.uniform(0, 1, (n, d))
        x = np.zeros((n, d))
        v_samp = [[None] * d for _ in range(d)]

        x[:, 0] = w[:, 0]
        v_samp[0][0] = w[:, 0]

        for i in range(1, d):
            v_samp[i][0] = w[:, i]

            for k in range(i - 1, -1, -1):
                edge = self.edges[k][i - k - 1]
                r = edge.get_r_predict(n)
                v_samp[i][0] = _clip_unit(
                    edge.copula.h_inverse(v_samp[i][0], v_samp[k][k], r))

            x[:, i] = v_samp[i][0]

            if i < d - 1:
                for j_idx in range(i):
                    edge = self.edges[j_idx][i - j_idx - 1]
                    r = edge.get_r_predict(n)
                    v_samp[i][j_idx + 1] = _clip_unit(
                        edge.copula.h(v_samp[i][j_idx], v_samp[j_idx][j_idx], r))

        return x

    # ── Summary ───────────────────────────────────────────────────

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
                if edge.method == 'MLE':
                    param = f"r={edge.fit_result.copula_param:.4f}"
                    ll = edge.fit_result.log_likelihood
                else:
                    param = f"alpha={edge.fit_result.alpha}"
                    ll = edge.fit_result.log_likelihood
                total_ll += ll
                rot_str = f" rot={rot}" if rot != 0 else ""
                print(f"  Edge {edge.idx}: {name}{rot_str}, {param}, logL={ll:.2f}")
        print(f"\nTotal logL (sum of edges): {total_ll:.2f}")