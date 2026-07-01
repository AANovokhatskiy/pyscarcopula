"""Equicorrelation Gaussian copula."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from pyscarcopula._types import DEFAULT_CONFIG, NumericalConfig
from pyscarcopula.copula.base import CopulaCapabilities
from pyscarcopula.copula.multivariate.base import MultivariateCopula
from pyscarcopula.copula.multivariate.conditional import (
    sample_gaussian_conditional,
    validate_multivariate_given,
)


_LBFGSB_FIT_KEYS = (
    "gtol",
    "ftol",
    "maxfun",
    "maxiter",
    "maxls",
    "eps",
    "maxcor",
    "finite_diff_rel_step",
)


class EquicorrGaussianCopula(MultivariateCopula):
    """Gaussian copula controlled by one equicorrelation parameter."""

    _capabilities = CopulaCapabilities(
        supports_gas=True,
        supports_scar_ou=True,
        supports_latent_grid=True,
        supports_conditional_sampling=True,
        has_dynamic_scalar_parameter=True,
    )

    def __init__(self, d, rotate=0):
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}")
        super().__init__(
            dimension=d, name=f"Equicorr Gaussian copula (d={d})")
        self._d = d
        self._bounds = [(-10.0, 10.0)]

    @property
    def d(self):
        return self._d

    def transform(self, x):
        from pyscarcopula.numerical import multivariate_native
        return multivariate_native.transform(self, x)

    def inv_transform(self, r):
        from pyscarcopula.numerical import multivariate_native
        return multivariate_native.inverse_transform(self, r)

    def dtransform(self, x):
        from pyscarcopula.numerical import multivariate_native
        return multivariate_native.dtransform(self, x)

    def log_likelihood(self, u, r=None):
        if r is None:
            from pyscarcopula._types import MLEResult
            if isinstance(self.fit_result, MLEResult):
                r = self.fit_result.copula_param
            else:
                r = float(self.transform(
                    np.array([self.fit_result.params.mu]))[0])
        from pyscarcopula.numerical import static_likelihood
        return static_likelihood.prepare(self, u).log_likelihood(float(r))

    def log_pdf_rows(self, u, r, t_index=None):
        from pyscarcopula.numerical import multivariate_native
        values, _ = multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index)
        return values

    def dlog_pdf_dr_rows(self, u, r, t_index=None):
        from pyscarcopula.numerical import multivariate_native
        _, values = multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index)
        return values

    def log_pdf_and_dlog_dr_rows(self, u, r, t_index=None):
        from pyscarcopula.numerical import multivariate_native
        return multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index)

    def pdf_on_grid(self, u_row, z_grid):
        values, _ = self.pdf_and_grad_on_grid_batch(
            np.asarray(u_row, dtype=np.float64)[None, :], z_grid)
        return values[0]

    def pdf_and_grad_on_grid(self, u_row, z_grid):
        values, gradients = self.pdf_and_grad_on_grid_batch(
            np.asarray(u_row, dtype=np.float64)[None, :], z_grid)
        return values[0], gradients[0]

    def pdf_and_grad_on_grid_batch(self, u, x_grid):
        from pyscarcopula.numerical import multivariate_native
        return multivariate_native.pdf_and_grad_grid(self, u, x_grid)

    def copula_grid_batch(self, u, x_grid):
        values, _ = self.pdf_and_grad_on_grid_batch(u, x_grid)
        return values

    def _fit_mle(
            self,
            u,
            config: NumericalConfig | None = None,
            gtol=None,
            ftol=None,
            maxfun=None,
            maxiter=None,
            maxls=None,
            eps=None,
            maxcor=None,
            finite_diff_rel_step=None):
        from pyscarcopula._types import MultivariateMLEResult

        config = config or DEFAULT_CONFIG
        optimizer_options = config.equicorr_optimizer.options(
            gtol=gtol,
            ftol=ftol,
            maxfun=maxfun,
            maxiter=maxiter,
            maxls=maxls,
            eps=eps,
            maxcor=maxcor,
            finite_diff_rel_step=finite_diff_rel_step,
        )
        from pyscarcopula.numerical import static_likelihood
        evaluator = static_likelihood.prepare(self, u)

        def neg_ll_and_grad(x):
            rho = self.transform(np.array([x[0]]))[0]
            value, grad_rho = evaluator.objective_and_gradient(
                rho, fail_value=config.fail_value)
            gradient = grad_rho * self.dtransform(
                np.array([x[0]], dtype=np.float64))
            return value, gradient

        result = minimize(
            neg_ll_and_grad,
            np.array([0.5]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-8.0, 8.0)],
            options=optimizer_options,
        )
        rho_hat = self.transform(result.x)[0]
        correlation = (
            (1.0 - rho_hat) * np.eye(self.d)
            + rho_hat * np.ones((self.d, self.d))
        )
        fitted = MultivariateMLEResult(
            log_likelihood=-result.fun,
            method="MLE",
            copula_name=self._name,
            success=result.success,
            nfev=result.nfev,
            message=str(getattr(result, "message", "")),
            copula_param=rho_hat,
            parameter_count=1,
            n_observations=len(u),
            model_parameters={"rho": rho_hat},
            correlation_matrix=correlation.copy(),
            diagnostics={
                "model_score": "not_applicable",
                "optimizer_gradient": "analytical",
                "gradient_kind": "analytical_chain_rule",
                "setup_derivative": "not_applicable",
                "filter_derivative": "not_applicable",
                "parameter_gradient": "analytical_rho",
                "transform_chain_rule": True,
                "corr_matrix": correlation.copy(),
            },
        )
        self.fit_result = fitted
        return fitted

    def fit(self, data, method="scar-tm-ou", to_pobs=False, **kwargs):
        from pyscarcopula._utils import pobs

        config = kwargs.pop("config", None)
        if "tol" in kwargs:
            raise TypeError("tol is not supported; use gtol")
        observations = np.asarray(data, dtype=np.float64)
        if to_pobs:
            observations = pobs(observations)
        self._last_u = observations

        if method.upper() == "MLE":
            optimizer_kwargs = {
                key: kwargs.pop(key)
                for key in _LBFGSB_FIT_KEYS
                if key in kwargs
            }
            return self._fit_mle(
                observations, config=config, **optimizer_kwargs)

        from pyscarcopula.api import fit
        result = fit(
            self, observations, method=method, config=config, **kwargs)
        self.fit_result = result
        return result

    def sample_at_parameter(self, n, r, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        parameters = np.atleast_1d(np.asarray(r, dtype=np.float64)).ravel()
        if parameters.size == 1:
            parameters = np.full(n, parameters[0], dtype=np.float64)
        elif parameters.size != n:
            raise ValueError(
                f"r must be scalar or array of length {n}, "
                f"got {parameters.size}")

        normal = rng.standard_normal((n, self._d))
        if np.all(parameters >= 0.0):
            common = rng.standard_normal((n, 1))
            values = (
                np.sqrt(1.0 - parameters)[:, None] * normal
                + np.sqrt(parameters)[:, None] * common
            )
        else:
            values = np.empty((n, self._d), dtype=np.float64)
            for index, rho in enumerate(parameters):
                correlation = (
                    (1.0 - rho) * np.eye(self._d)
                    + rho * np.ones((self._d, self._d))
                )
                values[index] = normal[index] @ np.linalg.cholesky(
                    correlation).T
        return norm.cdf(values)

    def sample(self, n, u=None, rng=None):
        """Generate observations reproducing the fitted model."""
        if self.fit_result is None:
            raise ValueError("Fit first")
        from pyscarcopula.api import sample as _api_sample

        u_data = u if u is not None else getattr(self, "_last_u", None)
        if u_data is None:
            raise ValueError(
                "No data for sample. "
                "Either call fit() first or pass u= explicitly.")
        return _api_sample(self, u_data, self.fit_result, n, rng=rng)

    def sample_conditional(self, n, r=None, given=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        given = validate_multivariate_given(given, self._d)
        if not given:
            if r is None:
                return self.sample(n, rng=rng)
            return self.sample_at_parameter(n, r=r, rng=rng)
        if r is None:
            r = self.fit_result.copula_param if self.fit_result else 0.5
        return sample_gaussian_conditional(
            n, self._d, r, given=given, rng=rng)

    def predict(
            self,
            n,
            u=None,
            rng=None,
            given=None,
            horizon="next",
            predictive_r_mode=None,
            predict_config=None):
        if predict_config is not None:
            from pyscarcopula.api import _resolve_predict_config
            config = _resolve_predict_config(
                predict_config, given, horizon, {
                    "predictive_r_mode": predictive_r_mode,
                })
            given = config.given
            horizon = config.horizon
            predictive_r_mode = config.predictive_r_mode
        if self.fit_result is None:
            raise ValueError("Fit first")
        if rng is None:
            rng = np.random.default_rng()

        from pyscarcopula._types import MLEResult
        if isinstance(self.fit_result, MLEResult):
            return self.sample_conditional(
                n, r=self.fit_result.copula_param, given=given, rng=rng)

        observations = u if u is not None else getattr(self, "_last_u", None)
        if observations is not None:
            grid, probability = self.xT_distribution(observations)
            indices = rng.choice(len(grid), size=n, p=probability)
            parameters = self.transform(grid[indices])
            return self.sample_conditional(
                n, r=parameters, given=given, rng=rng)

        kappa, mu, nu = self.fit_result.params.values
        variance = nu ** 2 / (2.0 * kappa)
        state = rng.normal(mu, np.sqrt(variance))
        parameter = self.transform(np.array([state]))[0]
        return self.sample_conditional(
            n, r=parameter, given=given, rng=rng)

    def predictive_mean(self, u):
        if self.fit_result is None:
            raise ValueError("Fit with SCAR first")
        kappa, mu, nu = self.fit_result.params.values
        from pyscarcopula.numerical import _cpp_scar_ou
        return _cpp_scar_ou.predictive_mean(kappa, mu, nu, u, self)

    def xT_distribution(self, u, K=300, grid_range=5.0):
        if self.fit_result is None:
            raise ValueError("Fit with SCAR first")
        kappa, mu, nu = self.fit_result.params.values
        from pyscarcopula.numerical import _cpp_scar_ou
        from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
        return _cpp_scar_ou.state_distribution(
            kappa,
            mu,
            nu,
            u,
            self,
            AutoTMConfig(K=K, grid_range=grid_range),
        )
