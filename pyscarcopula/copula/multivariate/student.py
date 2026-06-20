"""Static multivariate Student-t copula."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import (
    multivariate_t,
    t as t_dist,
)

from pyscarcopula._utils import clip_pseudo_observations, pobs
from pyscarcopula._types import MultivariateMLEResult
from pyscarcopula.copula.base import CopulaCapabilities
from pyscarcopula.copula.multivariate.base import MultivariateCopula
from pyscarcopula.copula.multivariate.corr_param import (
    estimate_kendall_correlation,
)


class StudentCopula(MultivariateCopula):
    """d-dimensional Student-t copula with fitted shape and degrees of freedom.

    Static MLE optimizes ``df`` directly in natural degrees-of-freedom units;
    no latent-state transform is used.
    """

    _capabilities = CopulaCapabilities()

    def __init__(self):
        super().__init__(name="Student-t copula")
        self.shape = None
        self.df = None
        self.correlation_preprocessing = None

    def fit(self, data, to_pobs=False, **kwargs):
        u = np.asarray(data, dtype=np.float64)
        if to_pobs:
            u = pobs(u)

        d = u.shape[1]
        self._set_dimension(d, allow_change=True)
        self.correlation_preprocessing = estimate_kendall_correlation(u)
        self.shape = self.correlation_preprocessing.correlation
        from pyscarcopula.numerical import static_likelihood
        evaluator = static_likelihood.prepare(self, u)

        def nll_profile(df_arr):
            return evaluator.objective_and_gradient(
                float(np.atleast_1d(df_arr)[0]))

        result = minimize(
            nll_profile,
            np.array([max(float(d), 5.0)]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(2.001, np.inf)],
            options={"gtol": 1e-2, "eps": 1e-4},
        )

        self.df = float(result.x[0])
        parameter_count = d * (d - 1) // 2 + 1
        fit_result = MultivariateMLEResult(
            log_likelihood=-float(result.fun),
            method="MLE",
            copula_name=self.name,
            success=bool(result.success),
            nfev=int(getattr(result, "nfev", 0)),
            message=str(getattr(result, "message", "")),
            copula_param=self.df,
            parameter_count=parameter_count,
            n_observations=len(u),
            model_parameters={
                "df": self.df,
                "correlation_matrix": self.shape.copy(),
            },
            correlation_matrix=self.shape,
            diagnostics={
                "model_score": "not_applicable",
                "optimizer_gradient": "analytical",
                "gradient_kind": "analytical",
                "setup_derivative": "not_applicable",
                "filter_derivative": "not_applicable",
                "df_gradient": "analytical",
                "corr_matrix": self.shape.copy(),
                **self.correlation_preprocessing.diagnostics(),
            },
        )
        self.fit_result = fit_result
        self._last_u = u
        return fit_result

    def _nll_for_df(self, u, df):
        u_c = clip_pseudo_observations(u)
        x = t_dist.ppf(u_c, df=df)
        R = estimate_kendall_correlation(x).correlation
        return self._nll_with_params(u, R, df)

    def _nll_with_params(self, u, R, df):
        try:
            from pyscarcopula.numerical import static_likelihood
            value, _ = static_likelihood.prepare_student(
                R, u).objective_and_gradient(df)
            return value
        except Exception:
            return 1e10

    def _nll(self, u):
        return self._nll_with_params(u, self.shape, self.df)

    def log_pdf_rows(self, u, parameter=None, **kwargs):
        from pyscarcopula.numerical import static_likelihood
        df = self.df if parameter is None else float(parameter)
        return static_likelihood.prepare(self, u).log_pdf_rows(df)

    def log_likelihood(self, u):
        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(self, u).log_likelihood(self.df)

    def _fitted_parameters(self):
        result = self.fit_result
        if isinstance(result, MultivariateMLEResult):
            return result.correlation_matrix, float(result.copula_param)
        return self.shape, self.df

    def sample(self, n, u=None, rng=None):
        correlation, df = self._fitted_parameters()
        if correlation is None or df is None:
            raise ValueError("Fit first")
        if rng is None:
            rng = np.random.default_rng()

        d = correlation.shape[0]
        x = multivariate_t.rvs(
            loc=np.zeros(d),
            shape=correlation,
            df=df,
            size=n,
            random_state=rng,
        )
        return t_dist.cdf(x, df=df)

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
                "Conditional prediction is not implemented for StudentCopula")
        return self.sample(n, u=u, rng=rng)
