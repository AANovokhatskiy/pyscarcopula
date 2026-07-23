"""Static multivariate Gaussian copula."""

import numpy as np
from scipy.stats import multivariate_normal, norm

from pyscarcopula._utils import clip_pseudo_observations, pobs
from pyscarcopula._types import MultivariateMLEResult
from pyscarcopula.copula.base import CopulaCapabilities
from pyscarcopula.copula.multivariate.base import (
    MultivariateCopula,
    model_state_locked,
)
from pyscarcopula.copula.multivariate.corr_param import validate_corr_matrix
from pyscarcopula.copula.multivariate.factor_correlation import (
    FactorCorrelation,
)
from pyscarcopula.copula.multivariate.factor_estimation import (
    estimate_factor_loadings,
)


def _validate_gaussian_fit_data(u):
    if u.ndim != 2:
        raise ValueError("data must have shape (n_observations, dimension)")
    if u.shape[0] == 0:
        raise ValueError("data must contain at least one observation")
    if u.shape[1] < 2:
        raise ValueError("data must contain at least two variables")
    if not np.all(np.isfinite(u)):
        raise ValueError("data must contain only finite values")


def _gaussian_score_correlation(u):
    u_c = clip_pseudo_observations(u)
    x = norm.ppf(u_c)
    if np.any(np.std(x, axis=0) <= 0.0):
        raise ValueError("data columns must not be constant")
    corr = np.corrcoef(x.T)
    corr = np.asarray(corr, dtype=np.float64)
    if corr.shape != (u.shape[1], u.shape[1]):
        raise ValueError("fitted correlation matrix has invalid shape")
    if not np.all(np.isfinite(corr)):
        raise ValueError(
            "fitted correlation matrix must contain only finite values")
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    validate_corr_matrix(corr)
    return corr


def _positive_integer(name, value, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validated_n_threads(value):
    value = _positive_integer("n_threads", value)
    if value > 256:
        raise ValueError("n_threads must be an integer in [1, 256]")
    return value


def _validated_budget(memory_budget_bytes, required, guidance):
    if memory_budget_bytes is None:
        return
    budget = _positive_integer(
        "memory_budget_bytes", memory_budget_bytes, allow_zero=True)
    if budget < int(required):
        raise MemoryError(
            f"sampling requires approximately {int(required)} bytes; "
            f"{guidance}")


class GaussianCopula(MultivariateCopula):
    """Static Gaussian copula with dense or compact factor correlation."""

    _capabilities = CopulaCapabilities(
        supports_conditional_sampling=True,
    )

    def __init__(
            self,
            d=None,
            *,
            corr_mode="dense",
            factor_rank=None,
            factor_loadings=None,
            factor_tile_size=16384,
            factor_uniqueness_min=1e-8,
            factor_seed=0,
            factor_oversampling=8):
        corr_mode = str(corr_mode).lower()
        if corr_mode not in {"dense", "factor"}:
            raise ValueError("corr_mode must be 'dense' or 'factor'")
        if corr_mode == "factor":
            if (
                    isinstance(factor_rank, (bool, np.bool_))
                    or not isinstance(factor_rank, (int, np.integer))):
                raise TypeError(
                    "factor_rank must be an integer in factor mode")
            factor_rank = int(factor_rank)
            if d is None:
                if factor_loadings is None:
                    raise ValueError(
                        "d is required when corr_mode='factor'")
                d = np.asarray(factor_loadings).shape[0]
            if not 1 <= factor_rank < int(d):
                raise ValueError("factor_rank must satisfy 1 <= k < d")
        elif factor_rank is not None or factor_loadings is not None:
            raise ValueError(
                "factor_rank and factor_loadings require "
                "corr_mode='factor'")

        super().__init__(dimension=d, name="Gaussian copula")
        self.corr = None
        self._corr_mode = corr_mode
        self._factor_rank = factor_rank
        self._factor_loadings = None
        self._factor_correlation = None
        self._factor_operator = None
        self._factor_initialization_diagnostics = {}
        self._constructor_factor_loadings = None
        self._factor_tile_size = _positive_integer(
            "factor_tile_size", factor_tile_size)
        self._factor_uniqueness_min = float(factor_uniqueness_min)
        if not (
                np.isfinite(self._factor_uniqueness_min)
                and 0.0 < self._factor_uniqueness_min < 1.0):
            raise ValueError(
                "factor_uniqueness_min must be finite and in (0, 1)")
        self._factor_seed = _positive_integer(
            "factor_seed", factor_seed, allow_zero=True)
        self._factor_oversampling = _positive_integer(
            "factor_oversampling", factor_oversampling, allow_zero=True)
        if factor_loadings is not None:
            self._set_factor_loadings(
                factor_loadings, diagnostics={"source": "supplied"})
            self._constructor_factor_loadings = (
                self._factor_loadings.copy())

    @property
    def corr_mode(self):
        return self._corr_mode

    @property
    def factor_rank(self):
        return self._factor_rank

    @property
    def factor_tile_size(self):
        return self._factor_tile_size

    @property
    def factor_loadings_(self):
        if self._factor_loadings is None:
            return None
        return self._factor_loadings.copy()

    @property
    def factor_uniqueness_(self):
        if self._factor_correlation is None:
            return None
        return self._factor_correlation.uniqueness.copy()

    @property
    def correlation_operator_(self):
        if self._corr_mode != "factor":
            raise AttributeError(
                "correlation_operator_ is only available in factor mode")
        if self._factor_operator is None:
            raise ValueError(
                "factor correlation is not initialized; call fit() or "
                "initialize_factor()")
        return self._factor_operator

    def to_correlation_matrix(
            self, *, max_dimension=2048, memory_budget_bytes=None):
        if self._corr_mode == "factor":
            return self.correlation_operator_.to_dense(
                max_dimension=max_dimension,
                memory_budget_bytes=memory_budget_bytes,
            )
        if self.corr is None:
            raise ValueError("Fit first")
        return self.corr.copy()

    def __getstate__(self):
        state = super().__getstate__()
        state["_factor_correlation"] = None
        state["_factor_operator"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._corr_mode = getattr(self, "_corr_mode", "dense")
        self._factor_correlation = None
        self._factor_operator = None
        loadings = getattr(self, "_factor_loadings", None)
        if self._corr_mode == "factor" and loadings is not None:
            self._set_factor_loadings(
                loadings,
                diagnostics=getattr(
                    self,
                    "_factor_initialization_diagnostics",
                    {"source": "restored"},
                ),
            )

    def _set_factor_loadings(self, loadings, *, diagnostics=None):
        if self._corr_mode != "factor":
            raise ValueError(
                "factor loadings require corr_mode='factor'")
        loadings = np.asarray(loadings, dtype=np.float64)
        expected = (self.dimension, self._factor_rank)
        if loadings.shape != expected:
            raise ValueError(
                f"factor_loadings must have shape {expected}, "
                f"got {loadings.shape}")
        factor = FactorCorrelation(
            loadings,
            uniqueness_min=self._factor_uniqueness_min,
            diagnostics={} if diagnostics is None else diagnostics,
        )
        self._factor_loadings = factor.loadings
        self._factor_correlation = factor
        self._factor_operator = factor.prepare()
        self._factor_initialization_diagnostics = dict(
            factor.diagnostics)

    @model_state_locked
    def initialize_factor(self, data, *, to_pobs=False):
        if self._corr_mode != "factor":
            raise ValueError(
                "initialize_factor requires corr_mode='factor'")
        u = np.asarray(data, dtype=np.float64)
        _validate_gaussian_fit_data(u)
        if u.shape[1] != self.dimension:
            raise ValueError(
                f"data must have {self.dimension} columns")
        if to_pobs:
            u = pobs(u)
        elif np.any((u < 0.0) | (u > 1.0)):
            raise ValueError(
                "factor initialization expects pseudo-observations "
                "in [0, 1]; use to_pobs=True")
        if self._factor_operator is not None:
            return self._factor_operator
        loadings, diagnostics = estimate_factor_loadings(
            u,
            self._factor_rank,
            uniqueness_min=self._factor_uniqueness_min,
            dimension_tile=self._factor_tile_size,
            seed=self._factor_seed,
            oversampling=self._factor_oversampling,
        )
        self._set_factor_loadings(loadings, diagnostics=diagnostics)
        return self._factor_operator

    def factor_diagnostics(self):
        if self._corr_mode != "factor":
            return {}
        identifiable = (
            self.dimension * self._factor_rank
            - self._factor_rank * (self._factor_rank - 1) // 2
        )
        diagnostics = {
            "corr_mode": "factor",
            "factor_rank": self._factor_rank,
            "factor_n_params": identifiable,
            "factor_tile_size": self._factor_tile_size,
            "factor_uniqueness_min": self._factor_uniqueness_min,
            "factor_initialized": self._factor_operator is not None,
            "representation": "factor_woodbury",
        }
        if self._factor_operator is not None:
            diagnostics.update(dict(self._factor_operator.diagnostics))
            diagnostics.update({
                f"initialization_{key}": value
                for key, value
                in self._factor_correlation.diagnostics.items()
            })
        return diagnostics

    @model_state_locked
    def fit(
            self,
            data,
            to_pobs=False,
            method='mle',
            config=None,
            **kwargs):
        """Fit the correlation matrix in Gaussian score space.

        Only ``method='mle'`` is supported: this is a static model without
        a dynamic scalar parameter.
        """
        if str(method).upper() != 'MLE':
            raise ValueError(
                f"GaussianCopula supports only method='mle', "
                f"got {method!r}")
        u = np.asarray(data, dtype=np.float64)
        _validate_gaussian_fit_data(u)
        if to_pobs:
            u = pobs(u)
            _validate_gaussian_fit_data(u)
        if self.dimension is not None and u.shape[1] != self.dimension:
            raise ValueError(
                f"data must have {self.dimension} columns")

        n_threads = _validated_n_threads(
            1 if config is None else config.n_threads)
        if self._corr_mode == "factor":
            if self._factor_operator is None:
                loadings, initialization = estimate_factor_loadings(
                    u,
                    self._factor_rank,
                    uniqueness_min=self._factor_uniqueness_min,
                    dimension_tile=self._factor_tile_size,
                    seed=self._factor_seed,
                    oversampling=self._factor_oversampling,
                )
                candidate = GaussianCopula(
                    d=u.shape[1],
                    corr_mode="factor",
                    factor_rank=self._factor_rank,
                    factor_loadings=loadings,
                    factor_tile_size=self._factor_tile_size,
                    factor_uniqueness_min=self._factor_uniqueness_min,
                    factor_seed=self._factor_seed,
                    factor_oversampling=self._factor_oversampling,
                )
                candidate._factor_initialization_diagnostics = initialization
                candidate._factor_correlation = FactorCorrelation(
                    candidate._factor_loadings,
                    uniqueness_min=self._factor_uniqueness_min,
                    diagnostics=initialization,
                )
                candidate._factor_operator = (
                    candidate._factor_correlation.prepare())
            else:
                candidate = GaussianCopula(
                    d=u.shape[1],
                    corr_mode="factor",
                    factor_rank=self._factor_rank,
                    factor_loadings=self._factor_loadings,
                    factor_tile_size=self._factor_tile_size,
                    factor_uniqueness_min=self._factor_uniqueness_min,
                    factor_seed=self._factor_seed,
                    factor_oversampling=self._factor_oversampling,
                )
                candidate._factor_initialization_diagnostics = dict(
                    self._factor_initialization_diagnostics)

            log_likelihood = candidate.log_likelihood(
                u, n_threads=n_threads)
            self._set_dimension(u.shape[1], allow_change=False)
            self._set_factor_loadings(
                candidate._factor_loadings,
                diagnostics=candidate._factor_initialization_diagnostics,
            )
            self.corr = None
            parameter_count = (
                self.dimension * self._factor_rank
                - self._factor_rank * (self._factor_rank - 1) // 2
            )
            diagnostics = {
                "estimator": "factor_gaussian_score_correlation",
                "corr_matrix": None,
                "n_threads": n_threads,
                **self.factor_diagnostics(),
            }
            result = MultivariateMLEResult(
                log_likelihood=log_likelihood,
                method="MLE",
                copula_name=self.name,
                success=True,
                message="two-stage factor Gaussian score correlation",
                copula_param=None,
                parameter_count=parameter_count,
                n_observations=len(u),
                model_parameters={
                    "corr_mode": "factor",
                    "factor_loadings": self.factor_loadings_,
                    "factor_uniqueness": self.factor_uniqueness_,
                    "factor_rank": self._factor_rank,
                },
                correlation_matrix=None,
                diagnostics=diagnostics,
            )
            self.fit_result = result
            self._last_u = u
            return result

        corr = _gaussian_score_correlation(u)
        candidate = GaussianCopula()
        candidate._set_dimension(u.shape[1], allow_change=True)
        candidate.corr = corr
        log_likelihood = -candidate._nll(u)

        self._set_dimension(u.shape[1], allow_change=True)
        self.corr = corr.copy()
        parameter_count = self.dimension * (self.dimension - 1) // 2
        result = MultivariateMLEResult(
            log_likelihood=log_likelihood,
            method="MLE",
            copula_name=self.name,
            success=True,
            message="closed-form Gaussian score correlation",
            copula_param=None,
            parameter_count=parameter_count,
            n_observations=len(u),
            model_parameters={
                "correlation_matrix": self.corr.copy(),
            },
            correlation_matrix=self.corr.copy(),
            diagnostics={
                "estimator": "gaussian_score_correlation",
                "corr_matrix": self.corr.copy(),
            },
        )
        self.fit_result = result
        self._last_u = u
        return result

    def log_likelihood(self, u, *, n_threads=1):
        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(
            self, u, n_threads=n_threads).log_likelihood(0.0)

    def log_pdf_rows(
            self, u, parameter=None, *, n_threads=1, **kwargs):
        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(
            self, u, n_threads=n_threads).log_pdf_rows(0.0)

    def _nll(self, u):
        return -self.log_likelihood(u)

    def _fitted_correlation(self):
        result = self.fit_result
        if (
                isinstance(result, MultivariateMLEResult)
                and result.correlation_matrix is not None):
            return result.correlation_matrix
        return self.corr

    def _factor_sampling_peak_bytes(self, rows, *, conditional=False):
        factor_width = 3 * self._factor_rank if conditional else (
            self._factor_rank)
        return int(rows) * (2 * self.dimension + factor_width + 4) * 8

    @model_state_locked
    def sample(
            self,
            n,
            u=None,
            rng=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        n = _positive_integer("n", n, allow_zero=True)
        n_threads = _validated_n_threads(n_threads)
        if self._corr_mode == "factor":
            operator = self.correlation_operator_
            _validated_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(n),
                "use sample_batches(), reduce batch_rows, or increase "
                "memory_budget_bytes",
            )
            latent = operator.sample_normal(
                n, rng=rng, n_threads=n_threads)
            return norm.cdf(latent)

        correlation = self._fitted_correlation()
        if correlation is None:
            raise ValueError("Fit first")
        if rng is None:
            rng = np.random.default_rng()

        d = correlation.shape[0]
        x = rng.multivariate_normal(np.zeros(d), correlation, size=n)
        return norm.cdf(x)

    @model_state_locked
    def sample_batches(
            self,
            n,
            u=None,
            rng=None,
            *,
            batch_rows=128,
            given=None,
            n_threads=1,
            memory_budget_bytes=None):
        n = _positive_integer("n", n, allow_zero=True)
        batch_rows = _positive_integer("batch_rows", batch_rows)
        n_threads = _validated_n_threads(n_threads)
        if rng is None:
            rng = np.random.default_rng()
        if given is not None:
            from pyscarcopula.copula.multivariate.conditional import (
                validate_multivariate_given,
            )
            given = validate_multivariate_given(given, self.dimension)
        if self._corr_mode == "factor":
            _validated_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(
                    min(n, batch_rows), conditional=bool(given)),
                "reduce batch_rows or increase memory_budget_bytes",
            )

        def blocks():
            for start in range(0, n, batch_rows):
                count = min(batch_rows, n - start)
                if given:
                    yield self.sample_conditional(
                        count,
                        given=given,
                        rng=rng,
                        n_threads=n_threads,
                        memory_budget_bytes=memory_budget_bytes,
                    )
                else:
                    yield self.sample(
                        count,
                        rng=rng,
                        n_threads=n_threads,
                        memory_budget_bytes=memory_budget_bytes,
                    )

        return blocks()

    @model_state_locked
    def sample_conditional(
            self,
            n,
            given,
            rng=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        """Sample conditionally with ``given={var_index: u_value}``."""
        n = _positive_integer("n", n, allow_zero=True)
        n_threads = _validated_n_threads(n_threads)
        if self._corr_mode == "factor":
            from pyscarcopula.copula.multivariate.conditional import (
                sample_factor_gaussian_conditional,
                validate_multivariate_given,
            )
            normalized = validate_multivariate_given(
                given, self.dimension)
            if not normalized:
                return self.sample(
                    n,
                    rng=rng,
                    n_threads=n_threads,
                    memory_budget_bytes=memory_budget_bytes,
                )
            _validated_budget(
                memory_budget_bytes,
                self._factor_sampling_peak_bytes(
                    n, conditional=True),
                "use sample_batches(), reduce batch_rows, or increase "
                "memory_budget_bytes",
            )
            return sample_factor_gaussian_conditional(
                n,
                self.correlation_operator_,
                normalized,
                rng=rng,
                n_threads=n_threads,
            )

        correlation = self._fitted_correlation()
        if correlation is None:
            raise ValueError("Fit first")
        from pyscarcopula.copula.multivariate.conditional import (
            sample_gaussian_copula_conditional,
        )
        return sample_gaussian_copula_conditional(
            n, correlation, given=given, rng=rng,
            n_threads=n_threads)

    @model_state_locked
    def predict(
            self,
            n,
            u=None,
            rng=None,
            given=None,
            horizon='next',
            predictive_r_mode=None,
            predict_config=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        if predict_config is not None:
            from pyscarcopula.api import _resolve_predict_config
            config = _resolve_predict_config(
                predict_config, given, horizon, {
                    "predictive_r_mode": predictive_r_mode,
                })
            given = config.given
        sampling_options = {}
        if n_threads != 1:
            sampling_options["n_threads"] = n_threads
        if memory_budget_bytes is not None:
            sampling_options["memory_budget_bytes"] = memory_budget_bytes
        if given:
            return self.sample_conditional(
                n,
                given=given,
                rng=rng,
                **sampling_options,
            )
        return self.sample(
            n,
            u=u,
            rng=rng,
            **sampling_options,
        )

    @model_state_locked
    def predict_batches(
            self,
            n,
            u=None,
            rng=None,
            *,
            batch_rows=128,
            given=None,
            horizon="next",
            predictive_r_mode=None,
            predict_config=None,
            n_threads=1,
            memory_budget_bytes=None):
        if predict_config is not None:
            from pyscarcopula.api import _resolve_predict_config
            config = _resolve_predict_config(
                predict_config, given, horizon, {
                    "predictive_r_mode": predictive_r_mode,
                })
            given = config.given
        return self.sample_batches(
            n,
            u=u,
            rng=rng,
            batch_rows=batch_rows,
            given=given,
            n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes,
        )
