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
    u_pair = np.column_stack((u1, u2))
    T = len(u1)

    best_copula = None
    best_result = None
    best_score = np.inf  # minimize AIC/BIC, or maximize logL

    for cop_class in candidates:
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

    if best_copula is None:
        raise RuntimeError("No copula could be fitted")

    return best_copula, best_result


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

    Tree structure (0-indexed):
        Tree j, edge i: copula for pair (variable j+1, variable j+i+2 | variables 1..j+1)

    The first variable is the root of tree 0, the second variable
    is the root of tree 1, etc.
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
            copulas=None, **kwargs):
        """
        Fit the C-vine copula.

        Parameters
        ----------
        data : (T, d) array
        method : str — 'mle' or SCAR method name
        to_pobs : bool
        copulas : None (auto-select) or list-of-lists of
                  (copula_class, rotation) tuples for manual specification.
                  copulas[j][i] = (GumbelCopula, 180)
        **kwargs : forwarded to copula.fit() for SCAR methods

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

                # Step 1: select copula (MLE always)
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

                # Step 2: if SCAR, refit with SCAR method
                if self.method != 'MLE':
                    scar_kwargs = {kk: vv for kk, vv in kwargs.items()
                                   if kk != 'alpha0'}
                    result = cop.fit(u_pair, method=method,
                                     alpha0=kwargs.get('alpha0'),
                                     **scar_kwargs)

                edge.copula = cop
                edge.fit_result = result
                edge.copula_param = (result.copula_param
                                     if self.method == 'MLE'
                                     else result.alpha)

                self.edges[j].append(edge)

            # Compute pseudo-obs for next tree level
            if j < d - 2:
                for i in range(n_edges):
                    u1 = _clip_unit(v[j][0])
                    u2 = _clip_unit(v[j][i + 1])
                    u_pair = np.column_stack((u1, u2))

                    edge = self.edges[j][i]
                    r = edge.get_r(u_pair)
                    # h(u2 | u1; r) — conditional CDF of u2 given u1
                    v[j + 1][i] = _clip_unit(
                        edge.copula.h(u2, u1, r))

        return self

    # ── Log-likelihood ────────────────────────────────────────────

    def log_likelihood(self, data, to_pobs=False):
        """
        Compute total log-likelihood of the C-vine.
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
                r = edge.get_r(u_pair)
                total_ll += np.sum(edge.copula.log_pdf(u1, u2, r))

            if j < d - 2:
                for i in range(n_edges):
                    u1 = _clip_unit(v[j][0])
                    u2 = _clip_unit(v[j][i + 1])
                    u_pair = np.column_stack((u1, u2))

                    edge = self.edges[j][i]
                    r = edge.get_r(u_pair)
                    v[j + 1][i] = _clip_unit(
                        edge.copula.h(u2, u1, r))

        return total_ll

    # ── Sampling (using fitted parameters on training data) ───────

    def sample(self, n, u_train=None):
        """
        Sample from fitted C-vine using Rosenblatt inverse.

        For SCAR: uses smoothed_params from u_train (must provide).
        For MLE: u_train not needed.

        Parameters
        ----------
        n : int — number of samples
        u_train : (T, d) or None — training data for SCAR smoothed params

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
                        r_here = r_cache[j][i]
                        v_train[j + 1][i] = _clip_unit(
                            edge.copula.h(u2, u1, r_here))

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
