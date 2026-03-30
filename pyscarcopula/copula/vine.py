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
from pyscarcopula._utils import pobs


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

    @property
    def method(self):
        if self.fit_result is None:
            return None
        return self.fit_result.method

    def get_r(self, u_pair, T=None):
        """
        Get copula parameter r for given data.
        MLE: constant (uses fit_result.copula_param).
        SCAR: smoothed_params via TM forward pass.
        """
        method = self.method
        if method is None:
            raise ValueError("Edge not fitted")

        if method.upper() == 'MLE':
            n = T if T is not None else len(u_pair)
            return np.full(n, self.fit_result.copula_param)
        else:
            alpha = _get_alpha(self.fit_result)
            from pyscarcopula.numerical.tm_functions import tm_forward_smoothed
            theta, mu, nu = alpha
            return tm_forward_smoothed(theta, mu, nu, u_pair, self.copula)

    def get_r_predict(self, n):
        """
        Get copula parameter r for prediction (sampling from x_T).
        MLE: constant (uses fit_result.copula_param).
        SCAR: sample from stationary OU.
        """
        method = self.method
        if method.upper() == 'MLE':
            return np.full(n, self.fit_result.copula_param)
        else:
            alpha = _get_alpha(self.fit_result)
            theta, mu, nu = alpha
            sigma2 = nu ** 2 / (2.0 * theta)
            x_T = np.random.normal(mu, np.sqrt(sigma2), n)
            return self.copula.transform(x_T)


def _get_alpha(fit_result):
    """Extract (theta, mu, nu) from fit_result."""
    return fit_result.params.values


def _get_gas_params(fit_result):
    """Extract (omega, alpha, beta) from fit_result."""
    return fit_result.params.values


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

def _kendall_tau(u1, u2):
    """Fast Kendall's tau via numpy (O(n log n) would be better, but this is simple)."""
    from scipy.stats import kendalltau
    tau, _ = kendalltau(u1, u2)
    return tau


def _itau_initial_param(cop_class, tau_abs, rotate):
    """Compute initial copula parameter from Kendall's tau.

    Returns parameter in the copula's natural domain (before inv_transform).
    Returns None if no closed-form available.
    """
    from pyscarcopula.copula.gumbel import GumbelCopula
    from pyscarcopula.copula.clayton import ClaytonCopula
    from pyscarcopula.copula.frank import FrankCopula
    from pyscarcopula.copula.joe import JoeCopula
    from pyscarcopula.copula.elliptical import BivariateGaussianCopula

    tau = max(tau_abs, 0.01)  # avoid division by zero

    if cop_class is GumbelCopula:
        # tau = 1 - 1/theta => theta = 1/(1-tau)
        theta = 1.0 / (1.0 - min(tau, 0.95))
        return max(theta, 1.001)

    if cop_class is ClaytonCopula:
        # tau = theta/(theta+2) => theta = 2*tau/(1-tau)
        theta = 2.0 * tau / (1.0 - min(tau, 0.95))
        return max(theta, 0.01)

    if cop_class is BivariateGaussianCopula:
        # tau = 2/pi * arcsin(rho) => rho = sin(pi*tau/2)
        rho = np.sin(np.pi * tau / 2.0)
        return np.clip(rho, -0.99, 0.99)

    if cop_class is FrankCopula:
        # Approximation: tau ≈ 1 - 4/theta + 4/theta^2 * (1-e^{-theta})
        # For moderate tau: theta ≈ 9*tau (rough but reasonable start)
        return max(9.0 * tau, 0.1)

    if cop_class is JoeCopula:
        # Rough approximation from Joe (1997):
        # tau ≈ 1 - 2/(theta*(theta-1)) for theta > 1
        # => theta ≈ 1 + 1/(1-tau) (crude)
        theta = 1.0 + 1.0 / (1.0 - min(tau, 0.9))
        return max(theta, 1.001)

    return None


def _rotation_compatible(tau, rotate):
    """Check if rotation is compatible with sign of Kendall's tau.

    Returns False only for clear incompatibility (strong tau
    with wrong rotation). Weak dependence passes all rotations.
    """
    # Weak dependence — don't prune, let AIC decide
    if abs(tau) < 0.15:
        return True

    if rotate == 0 or rotate == 180:
        return tau > 0
    else:  # 90, 270
        return tau < 0


def select_best_copula(u1, u2, candidates, allow_rotations=True,
                       criterion='aic', transform_type='xtanh'):
    """
    Select best bivariate copula for (u1, u2) by AIC/BIC/logL.

    Uses Kendall's tau for:
    1. Pre-screening: skip families/rotations incompatible with tau sign
    2. Initial point: itau(tau) gives a better x0 for L-BFGS-B

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
    from pyscarcopula._types import IndependentResult

    u_pair = np.column_stack((u1, u2))
    T = len(u1)

    # Compute Kendall's tau once
    tau = _kendall_tau(u1, u2)

    # Start with independence as baseline (AIC=0, BIC=0, logL=0)
    indep = IndependentCopula()
    indep_result = IndependentResult(
        log_likelihood=0.0, method='MLE',
        copula_name=indep.name, success=True)

    if criterion == 'aic':
        best_score = 0.0
    elif criterion == 'bic':
        best_score = 0.0
    else:
        best_score = 0.0

    best_copula = indep
    best_result = indep_result

    for cop_class in candidates:
        if cop_class is IndependentCopula:
            continue

        rotations = _all_rotations(cop_class) if allow_rotations else [0]

        for angle in rotations:
            # Pre-screening: skip incompatible rotation+tau
            if not _rotation_compatible(tau, angle):
                continue

            try:
                try:
                    cop = cop_class(rotate=angle, transform_type=transform_type)
                except TypeError:
                    cop = cop_class(rotate=angle)

                # Compute initial point from Kendall's tau
                tau_for_family = abs(tau)
                r0 = _itau_initial_param(cop_class, tau_for_family, angle)

                alpha0 = None
                if r0 is not None:
                    x0 = cop.inv_transform(np.atleast_1d(np.array([r0], dtype=np.float64)))
                    alpha0 = np.atleast_1d(x0)[0:1]

                from pyscarcopula.api import fit as _api_fit
                result = _api_fit(cop, u_pair, method='mle',
                                  alpha0=alpha0)
                logL = result.log_likelihood
                n_params = 1

                if criterion == 'aic':
                    score = -2 * logL + 2 * n_params
                elif criterion == 'bic':
                    score = -2 * logL + n_params * np.log(T)
                else:
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
        from pyscarcopula.numerical.gas_filter import gas_mixture_h
        alpha = _get_gas_params(edge.fit_result)
        scaling = getattr(edge.fit_result, 'scaling', 'unit')
        return gas_mixture_h(alpha[0], alpha[1], alpha[2],
                              u_pair, edge.copula, scaling)

    else:
        # SCAR-TM-OU: mixture h via transfer matrix
        from pyscarcopula.numerical.tm_functions import tm_forward_mixture_h as _tm_forward_mixture_h
        alpha = _get_alpha(edge.fit_result)
        theta, mu, nu = alpha
        return _tm_forward_mixture_h(theta, mu, nu, u_pair,
                                      edge.copula, K, grid_range)


# ══════════════════════════════════════════════════════════════════
# Helper: edge log-likelihood dispatch
# ══════════════════════════════════════════════════════════════════

def _edge_log_likelihood(edge, u_pair, K=300, grid_range=5.0):
    """
    Compute log-likelihood for one edge using the correct method.

    MLE:  sum log c(u1, u2; theta_mle)
    GAS:  sum log c(u1, u2; Psi(f_t))  (score-driven filter)
    SCAR: log integral (transfer matrix likelihood)
    """
    from pyscarcopula.copula.independent import IndependentCopula
    if isinstance(edge.copula, IndependentCopula):
        return 0.0

    method = edge.method.lower() if edge.method else 'mle'
    cop = edge.copula

    if method == 'mle':
        r = edge.fit_result.copula_param
        u1 = u_pair[:, 0]
        u2 = u_pair[:, 1]
        return np.sum(cop.log_pdf(u1, u2, np.full(len(u1), float(r))))

    elif method == 'gas':
        from pyscarcopula.numerical.gas_filter import gas_negloglik
        alpha = _get_gas_params(edge.fit_result)
        scaling = getattr(edge.fit_result, 'scaling', 'unit')
        return -gas_negloglik(alpha[0], alpha[1], alpha[2],
                              u_pair, cop, scaling)

    else:
        from pyscarcopula.numerical.tm_functions import tm_loglik
        alpha = _get_alpha(edge.fit_result)
        theta, mu, nu = alpha
        return -tm_loglik(theta, mu, nu, u_pair, cop, K, grid_range)


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
                    try:
                        cop = cop_class(rotate=rotation, transform_type=transform_type)
                    except TypeError:
                        cop = cop_class(rotate=rotation)
                    from pyscarcopula.api import fit as _api_fit
                    result = _api_fit(cop, u_pair, method='mle')
                else:
                    # Automatic selection via AIC/BIC
                    cop, result = select_best_copula(
                        u1, u2, self._get_candidates(),
                        self.allow_rotations, self.criterion,
                        transform_type=transform_type)

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
                    from pyscarcopula.api import fit as _api_fit
                    scar_kwargs = {kk: vv for kk, vv in kwargs.items()
                                   if kk != 'alpha0'}
                    result = _api_fit(cop, u_pair, method=method,
                                      alpha0=kwargs.get('alpha0'),
                                      **scar_kwargs)

                edge.copula = cop
                edge.fit_result = result

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

        # ── Build aggregate fit_result ─────────────────────────────
        from scipy.optimize import OptimizeResult

        total_ll = sum(e.fit_result.log_likelihood
                       for tree in self.edges for e in tree)
        total_nfev = sum(getattr(e.fit_result, 'nfev', 0)
                         for tree in self.edges for e in tree)
        n_edges = sum(len(tree) for tree in self.edges)

        self.fit_result = OptimizeResult()
        self.fit_result.log_likelihood = total_ll
        self.fit_result.method = method
        self.fit_result.name = f"C-vine ({d}d, {n_edges} edges)"
        self.fit_result.nfev = total_nfev
        self.fit_result.success = True
        self._last_u = u  # store for predict

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
                total_ll += _edge_log_likelihood(edge, u_pair, K, grid_range)

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

    # ── Sampling (reproduces fitted model) ───────────────────────

    def sample(self, n, u_train=None, K=300, grid_range=5.0):
        """
        Sample from fitted C-vine using Rosenblatt inverse.

        For MLE: constant r per edge.
        For SCAR-TM/SCAR-MC: OU trajectory per edge, r(t) = Psi(x(t)).
        For GAS: recursive simulation — at each step t, generate one
            observation, compute score, update f_{t+1}.

        fit(vine, sample(...)) should recover similar parameters.

        Parameters
        ----------
        n : int — number of samples
        u_train : ignored (kept for backward compatibility)
        K : int — ignored
        grid_range : float — ignored

        Returns
        -------
        x : (n, d) — samples in [0,1]^d
        """
        if self.edges is None:
            raise ValueError("Fit first")

        d = self.d
        rng = np.random.default_rng()

        from pyscarcopula._types import GASResult

        # Check if any edge uses GAS — need step-by-step simulation
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

        # Generate r trajectory for each edge ONCE
        r_sampling = [[None] * (d - 1) for _ in range(d - 1)]
        for j in range(d - 1):
            for i in range(d - j - 1):
                edge = self.edges[j][i]
                r_sampling[j][i] = self._generate_r_for_sample(
                    edge, n, rng)

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
                        edge.copula.h(v_samp[i][j_idx], v_samp[j_idx][j_idx], r))

        return x

    def _sample_stepwise(self, n, d, rng):
        """Step-by-step sample for GAS edges.

        At each time step t:
          1. Use current r_t per edge for Rosenblatt inverse → one obs
          2. Compute pseudo-obs for each edge from the generated obs
          3. Update GAS edges: f_{t+1} = omega + beta*f_t + alpha*score_t
        """
        from pyscarcopula._types import GASResult, LatentResult, MLEResult
        from pyscarcopula.copula.independent import IndependentCopula

        x = np.zeros((n, d))

        # Initialize r state for each edge
        n_trees = d - 1
        # r_state[j][i] = current scalar r for edge (j, i)
        r_state = [[None] * (d - 1) for _ in range(n_trees)]
        # f_state[j][i] = current f for GAS edges
        f_state = [[None] * (d - 1) for _ in range(n_trees)]
        # For SCAR: precompute full OU trajectories
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
                        edge.copula.transform(np.array([f_state[j][i]]))[0])
                elif isinstance(edge.fit_result, LatentResult):
                    # Precompute OU trajectory
                    r_ou_path[j][i] = self._generate_r_for_sample(
                        edge, n, rng)
                    r_state[j][i] = float(r_ou_path[j][i][0])

        score_eps = 1e-4

        for t in range(n):
            # Update r from OU paths
            for j in range(n_trees):
                for i in range(d - j - 1):
                    if r_ou_path[j][i] is not None:
                        r_state[j][i] = float(r_ou_path[j][i][t])

            # Generate one d-dimensional observation via Rosenblatt inverse
            w = rng.uniform(0, 1, d)
            # v_samp[k][k] = pivot pseudo-obs at tree level k
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

                # Compute h-transformed pseudo-obs for higher trees
                if i < d - 1:
                    for j_idx in range(i):
                        edge = self.edges[j_idx][i - j_idx - 1]
                        r_ji = r_state[j_idx][i - j_idx - 1]
                        v_samp[i][j_idx + 1] = float(_clip_unit(np.atleast_1d(
                            edge.copula.h(
                                np.array([v_samp[i][j_idx]]),
                                np.array([v_samp[j_idx][j_idx]]),
                                np.array([r_ji]))))[0])

            # Update GAS state using generated observation
            # Build pseudo-obs for each edge from x[t]
            v_obs = [[None] * d for _ in range(d)]
            for i in range(d):
                v_obs[0][i] = x[t, i]

            for j in range(n_trees):
                for i in range(d - j - 1):
                    edge = self.edges[j][i]
                    u1_t = float(_clip_unit(np.atleast_1d(v_obs[j][0]))[0])
                    u2_t = float(_clip_unit(np.atleast_1d(v_obs[j][i + 1]))[0])

                    # Update GAS state
                    if isinstance(edge.fit_result, GASResult):
                        p = edge.fit_result.params
                        f_t = f_state[j][i]
                        r_t = r_state[j][i]
                        scaling = getattr(edge.fit_result, 'scaling', 'unit')

                        u1a = np.array([u1_t])
                        u2a = np.array([u2_t])

                        f_plus = f_t + score_eps
                        f_minus = f_t - score_eps
                        r_plus = float(edge.copula.transform(np.array([f_plus]))[0])
                        r_minus = float(edge.copula.transform(np.array([f_minus]))[0])

                        ll_plus = float(edge.copula.log_pdf(u1a, u2a, np.array([r_plus]))[0])
                        ll_minus = float(edge.copula.log_pdf(u1a, u2a, np.array([r_minus]))[0])

                        nabla = (ll_plus - ll_minus) / (2.0 * score_eps)

                        if scaling == 'fisher':
                            ll_t = float(edge.copula.log_pdf(u1a, u2a, np.array([r_t]))[0])
                            d2 = (ll_plus - 2.0 * ll_t + ll_minus) / (score_eps ** 2)
                            fisher = max(-d2, 1e-6)
                            s_t = nabla / fisher
                        else:
                            s_t = nabla

                        s_t = np.clip(s_t, -100.0, 100.0)
                        f_new = p.omega + p.beta * f_t + p.alpha * s_t
                        f_new = np.clip(f_new, -50.0, 50.0)
                        f_state[j][i] = float(f_new)
                        r_state[j][i] = float(
                            edge.copula.transform(np.array([f_new]))[0])

                    # Compute h for pseudo-obs at next tree level
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
        """Alias for sample (interface compatibility with BivariateCopula)."""
        return self.sample(n)

    def _generate_r_for_sample(self, edge, n, rng):
        """Generate r trajectory for sample (model reproduction).

        MLE: constant r.
        SCAR: OU trajectory with dt = 1/(n-1).
        GAS: unconditional mean f_bar.
        """
        from pyscarcopula.copula.independent import IndependentCopula
        from pyscarcopula._types import LatentResult, MLEResult, GASResult

        if isinstance(edge.copula, IndependentCopula):
            return np.zeros(n)

        if isinstance(edge.fit_result, MLEResult):
            return np.full(n, edge.fit_result.copula_param)

        if isinstance(edge.fit_result, LatentResult):
            alpha = _get_alpha(edge.fit_result)
            theta, mu, nu = alpha
            dt = 1.0 / (n - 1) if n > 1 else 1.0
            rho_ou = np.exp(-theta * dt)
            sigma_cond = np.sqrt(
                nu ** 2 / (2.0 * theta) * (1.0 - rho_ou ** 2))
            x_path = np.empty(n)
            x_path[0] = rng.normal(mu, nu / np.sqrt(2.0 * theta))
            for t in range(1, n):
                x_path[t] = (mu + rho_ou * (x_path[t - 1] - mu)
                             + sigma_cond * rng.standard_normal())
            return edge.copula.transform(x_path)

        if isinstance(edge.fit_result, GASResult):
            p = edge.fit_result.params
            omega, _, beta = p.omega, p.alpha, p.beta
            if abs(beta) < 1.0 - 1e-8:
                f_bar = omega / (1.0 - beta)
            else:
                f_bar = omega
            r_bar = edge.copula.transform(np.array([f_bar]))[0]
            return np.full(n, r_bar)

        return edge.get_r_predict(n)

    # ── Prediction (conditional on fitted data) ────────────────

    def _generate_r_for_predict(self, edge, j, i, n,
                                v_train, K, grid_range):
        """Generate r for predict (next-step conditional).

        MLE: constant r.
        SCAR-TM: mixture sampling from posterior p(x_T | data).
        GAS: last filtered value f_T.
        """
        from pyscarcopula.copula.independent import IndependentCopula
        from pyscarcopula._types import LatentResult, MLEResult, GASResult

        if isinstance(edge.copula, IndependentCopula):
            return np.zeros(n)

        if isinstance(edge.fit_result, MLEResult):
            return np.full(n, edge.fit_result.copula_param)

        if isinstance(edge.fit_result, LatentResult):
            if v_train is not None:
                alpha = _get_alpha(edge.fit_result)
                theta, mu, nu = alpha
                u1 = _clip_unit(v_train[j][0])
                u2 = _clip_unit(v_train[j][i + 1])
                u_pair = np.column_stack((u1, u2))
                from pyscarcopula.numerical.tm_functions import tm_xT_distribution
                z_grid, prob = tm_xT_distribution(
                    theta, mu, nu, u_pair, edge.copula, K, grid_range)
                idx = np.random.choice(len(z_grid), size=n, p=prob)
                return edge.copula.transform(z_grid[idx])
            else:
                return edge.get_r_predict(n)

        if isinstance(edge.fit_result, GASResult):
            # Use cached r_last from fit (avoids expensive gas_filter rerun)
            r_last = getattr(edge.fit_result, 'r_last', None)
            if r_last is not None and r_last != 0.0:
                return np.full(n, r_last)
            # Fallback: run gas_filter if r_last not available
            if v_train is not None:
                from pyscarcopula.numerical.gas_filter import gas_filter
                p = edge.fit_result.params
                u1 = _clip_unit(v_train[j][0])
                u2 = _clip_unit(v_train[j][i + 1])
                u_pair = np.column_stack((u1, u2))
                scaling = getattr(edge.fit_result, 'scaling', 'unit')
                _, r_path, _ = gas_filter(
                    p.omega, p.alpha, p.beta,
                    u_pair, edge.copula, scaling)
                return np.full(n, r_path[-1])
            else:
                p = edge.fit_result.params
                f_bar = p.omega / (1.0 - p.beta) if abs(p.beta) < 0.999 else p.omega
                return np.full(n, edge.copula.transform(np.array([f_bar]))[0])

        return edge.get_r_predict(n)

    def predict(self, n, u=None, K=300, grid_range=5.0):
        """
        Conditional predict: sample from vine for next-step prediction.

        For MLE: constant parameter.
        For SCAR-TM: mixture sampling from posterior p(x_T | data)
            for each edge independently.
        For GAS: uses last filtered value f_T.

        Parameters
        ----------
        n : int — number of samples
        u : (T, d) or None — data for conditioning.
            If None, uses data from last fit() call.
        K : int — grid size for SCAR
        grid_range : float

        Returns
        -------
        x : (n, d) — samples in [0,1]^d
        """
        if self.edges is None:
            raise ValueError("Fit first")

        u_data = u if u is not None else getattr(self, '_last_u', None)

        d = self.d
        w = np.random.uniform(0, 1, (n, d))
        x = np.zeros((n, d))
        v_samp = [[None] * d for _ in range(d)]

        x[:, 0] = w[:, 0]
        v_samp[0][0] = w[:, 0]

        # Build v_train only if needed (SCAR-TM edges need it for
        # posterior computation on pseudo-obs at higher trees).
        # GAS edges use cached r_last and don't need v_train.
        v_train = None
        from pyscarcopula._types import LatentResult
        needs_v_train = any(
            isinstance(self.edges[j][i].fit_result, LatentResult)
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
                r_pred[j][i] = self._generate_r_for_predict(
                    edge, j, i, n, v_train, K, grid_range)

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
                if edge.method.upper() == 'MLE':
                    param = f"r={edge.fit_result.copula_param:.4f}"
                else:
                    alpha = _get_alpha(edge.fit_result)
                    param = f"alpha={alpha}"
                ll = edge.fit_result.log_likelihood
                total_ll += ll
                rot_str = f" rot={rot}" if rot != 0 else ""
                print(f"  Edge {edge.idx}: {name}{rot_str}, {param}, logL={ll:.2f}")
        print(f"\nTotal logL (sum of edges): {total_ll:.2f}")