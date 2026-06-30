"""Static multivariate Gaussian copula."""

import numpy as np
from scipy.stats import multivariate_normal, norm

from pyscarcopula._utils import clip_pseudo_observations, pobs
from pyscarcopula._types import MultivariateMLEResult
from pyscarcopula.copula.base import CopulaCapabilities
from pyscarcopula.copula.multivariate.base import MultivariateCopula


class GaussianCopula(MultivariateCopula):
    """d-dimensional Gaussian copula with a fitted correlation matrix."""

    _capabilities = CopulaCapabilities()

    def __init__(self):
        super().__init__(name="Gaussian copula")
        self.corr = None

    def fit(self, data, to_pobs=False, **kwargs):
        """Fit the correlation matrix in Gaussian score space."""
        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        u_c = clip_pseudo_observations(u)
        x = norm.ppf(u_c)
        self._set_dimension(u.shape[1], allow_change=True)
        self.corr = np.corrcoef(x.T)
        log_likelihood = -self._nll(u)
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

    def log_likelihood(self, u):
        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(self, u).log_likelihood(0.0)

    def log_pdf_rows(self, u, parameter=None, **kwargs):
        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(self, u).log_pdf_rows(0.0)

    def _nll(self, u):
        return -self.log_likelihood(u)

    def _fitted_correlation(self):
        result = self.fit_result
        if (
                isinstance(result, MultivariateMLEResult)
                and result.correlation_matrix is not None):
            return result.correlation_matrix
        return self.corr

    def sample(self, n, u=None, rng=None):
        correlation = self._fitted_correlation()
        if correlation is None:
            raise ValueError("Fit first")
        if rng is None:
            rng = np.random.default_rng()

        d = correlation.shape[0]
        x = rng.multivariate_normal(np.zeros(d), correlation, size=n)
        return norm.cdf(x)

    def predict(self, n, u=None, rng=None, given=None, horizon='next',
                predictive_r_mode=None, predict_config=None):
        if predict_config is not None:
            from pyscarcopula.api import _resolve_predict_config
            config = _resolve_predict_config(
                predict_config, given, horizon, {
                    "predictive_r_mode": predictive_r_mode,
                })
            given = config.given
        if given:
            raise NotImplementedError(
                "Conditional prediction is not implemented for GaussianCopula")
        return self.sample(n, u=u, rng=rng)
