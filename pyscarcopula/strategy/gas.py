"""GAS estimation strategy backed by the native numerical evaluator."""

import numpy as np
from scipy.optimize import Bounds, minimize

from pyscarcopula._types import (
    DEFAULT_CONFIG,
    GASResult,
    NumericalConfig,
    PredictiveState,
    gas_params,
)
from pyscarcopula.numerical import _cpp_gas
from pyscarcopula.numerical.gas_filter import (
    gas_filter,
    gas_loglik,
    gas_mixture_h,
    gas_negloglik,
    gas_predict_param,
)
from pyscarcopula.strategy._base import (
    copula_dimension,
    is_multivariate_copula,
    register_strategy,
)
from pyscarcopula.strategy.predict_helpers import sample_predictive


@register_strategy("GAS")
class GASStrategy:
    """GAS estimation strategy.

    Parameters
    ----------
    config : NumericalConfig
    scaling : {'unit', 'fisher'}
        Score scaling type. ``unit`` is recommended for production.

    Notes
    -----
    GAS numerical operations require the compiled extension. There is no
    Python numerical backend or silent fallback. The copula score driving the
    recursion is computed natively. L-BFGS-B receives only objective values,
    so its gradient with respect to ``(omega, gamma, beta)`` is numerical.
    """

    def __init__(
        self,
        config: NumericalConfig | None = None,
        scaling: str = "unit",
        **kwargs,
    ):
        if "backend" in kwargs:
            raise TypeError(
                "GAS backend selection was removed; native execution is "
                "always used")
        self.config = config or DEFAULT_CONFIG
        self.scaling = scaling

    def _score_eps(self, result: GASResult | None = None) -> float:
        if result is None:
            return float(self.config.gas_score_eps)
        return float(getattr(result, "score_eps", self.config.gas_score_eps))

    def _optimizer_config(self, copula):
        config_name = getattr(copula, "_gas_optimizer_config", None)
        if config_name is not None:
            return getattr(self.config, config_name)
        return self.config.gas_optimizer

    def fit(
        self,
        copula,
        u: np.ndarray,
        gamma0: np.ndarray | None = None,
        gtol: float | None = None,
        ftol: float | None = None,
        maxfun: int | None = None,
        maxiter: int | None = None,
        maxls: int | None = None,
        eps: float | None = None,
        maxcor: int | None = None,
        finite_diff_rel_step: float | None = None,
        score_eps: float | None = None,
        gamma_bound: float | None = None,
        beta_bound: float | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> GASResult:
        """Fit the native GAS model."""
        if "tol" in kwargs:
            raise TypeError("tol is not supported; use gtol")
        if "backend" in kwargs:
            raise TypeError(
                "GAS backend selection was removed; native execution is "
                "always used")
        corr_num_params = getattr(copula, "_corr_num_params", lambda: 0)()
        if corr_num_params:
            raise NotImplementedError(
                "joint static correlation estimation is implemented for "
                "MLE and SCAR-TM-OU, not GAS")
        _cpp_gas.ensure_supported(copula)
        _cpp_gas.require_available()

        optimizer_options = self._optimizer_config(copula).options(
            gtol=gtol,
            ftol=ftol,
            maxfun=maxfun,
            maxiter=maxiter,
            maxls=maxls,
            eps=eps,
            maxcor=maxcor,
            finite_diff_rel_step=finite_diff_rel_step,
        )
        score_eps = float(
            score_eps
            if score_eps is not None
            else self.config.gas_score_eps
        )
        gamma_bound = float(
            gamma_bound
            if gamma_bound is not None
            else self.config.gas_gamma_bound
        )
        beta_bound = float(
            beta_bound
            if beta_bound is not None
            else self.config.gas_beta_bound
        )
        if gamma_bound <= 0:
            raise ValueError("gamma_bound must be positive")
        if not 0 < beta_bound < 1:
            raise ValueError("beta_bound must be in (0, 1)")

        if gamma0 is None:
            from pyscarcopula.strategy.mle import MLEStrategy

            mle_result = MLEStrategy(config=self.config).fit(copula, u)
            mu_mle = float(
                np.atleast_1d(
                    copula.inv_transform(
                        np.atleast_1d(mle_result.copula_param)
                    )
                )[0]
            )
            gamma0 = np.array([mu_mle * 0.05, 0.05, 0.95])

        if verbose:
            print(
                f"GAS fit: gamma0={gamma0}, scaling={self.scaling}, "
                f"score_eps={score_eps}, options={optimizer_options}, "
                f"gamma_bound={gamma_bound}, beta_bound={beta_bound}"
            )

        bounds = Bounds(
            [-np.inf, -gamma_bound, -beta_bound],
            [np.inf, gamma_bound, beta_bound],
        )

        def objective(x):
            return gas_negloglik(
                x[0],
                x[1],
                x[2],
                u,
                copula,
                self.scaling,
                score_eps,
            )

        result = minimize(
            objective,
            gamma0,
            method="L-BFGS-B",
            bounds=bounds,
            options=optimizer_options,
        )
        params = gas_params(
            omega=result.x[0],
            gamma=result.x[1],
            beta=result.x[2],
            gamma_bound=gamma_bound,
            beta_bound=beta_bound,
        )

        success = bool(result.success)
        message = str(result.message)
        try:
            final_log_likelihood = gas_loglik(
                result.x[0],
                result.x[1],
                result.x[2],
                u,
                copula,
                self.scaling,
                score_eps,
            )
            if not np.isfinite(final_log_likelihood):
                raise FloatingPointError(
                    "final GAS log-likelihood is not finite")
            r_last = gas_predict_param(
                result.x[0],
                result.x[1],
                result.x[2],
                u,
                copula,
                self.scaling,
                score_eps,
            )
        except Exception as exc:
            success = False
            final_log_likelihood = -1e10
            r_last = 0.0
            message = f"{message}; final native GAS validation failed: {exc}"

        return GASResult(
            log_likelihood=final_log_likelihood,
            method="GAS",
            copula_name=copula.name,
            success=success,
            nfev=result.nfev,
            message=message,
            params=params,
            scaling=self.scaling,
            score_eps=score_eps,
            r_last=r_last,
            diagnostics={
                "model_score": "native",
                "optimizer_gradient": "numerical",
                "gradient_kind": "numerical_optimizer",
                "setup_derivative": "not_provided",
                "filter_derivative": "not_provided_to_optimizer",
                "analytical_grad_requested": False,
                "analytical_grad_used": False,
            },
        )

    def log_likelihood(self, copula, u: np.ndarray, result: GASResult) -> float:
        p = result.params
        return gas_loglik(
            p.omega,
            p.gamma,
            p.beta,
            u,
            copula,
            result.scaling,
            self._score_eps(result),
        )

    def predictive_mean(
        self,
        copula,
        u: np.ndarray,
        result: GASResult,
    ) -> np.ndarray:
        p = result.params
        _, r_path, _ = gas_filter(
            p.omega,
            p.gamma,
            p.beta,
            u,
            copula,
            result.scaling,
            self._score_eps(result),
        )
        return r_path

    def rosenblatt_e2(
        self,
        copula,
        u: np.ndarray,
        result: GASResult,
    ) -> np.ndarray:
        return self.mixture_h(copula, u, result)

    def mixture_h(
        self,
        copula,
        u: np.ndarray,
        result: GASResult,
    ) -> np.ndarray:
        if is_multivariate_copula(copula):
            raise NotImplementedError(
                "pair h-functions are not defined for multivariate GAS")
        p = result.params
        return gas_mixture_h(
            p.omega,
            p.gamma,
            p.beta,
            u,
            copula,
            result.scaling,
            self._score_eps(result),
        )

    def objective(
        self,
        copula,
        u: np.ndarray,
        gamma: np.ndarray,
        **kwargs,
    ) -> float:
        if "backend" in kwargs:
            raise TypeError(
                "GAS backend selection was removed; native execution is "
                "always used")
        score_eps = float(kwargs.get("score_eps", self._score_eps()))
        return gas_negloglik(
            gamma[0],
            gamma[1],
            gamma[2],
            u,
            copula,
            self.scaling,
            score_eps,
        )

    def sample(self, copula, u, result, n, rng=None, **kwargs):
        """Recursively sample using native GAS state updates."""
        if rng is None:
            rng = np.random.default_rng()
        p = result.params
        score_eps = self._score_eps(result)
        d = copula_dimension(copula, u)
        if d is None:
            raise ValueError("copula dimension is unknown")

        state = _cpp_gas.initial_state(
            p.omega,
            p.gamma,
            p.beta,
            copula,
            result.scaling,
            score_eps,
        )
        g_t = state.g
        r_t = state.parameter
        samples = np.empty((n, d), dtype=np.float64)
        multivariate = is_multivariate_copula(copula)

        for t in range(n):
            if multivariate:
                obs = sample_predictive(
                    copula,
                    1,
                    np.array([r_t]),
                    rng=rng,
                    d=d,
                )
            else:
                obs = copula.sample_at_parameter(
                    1, np.array([r_t]), rng=rng)
            samples[t] = obs[0]
            if t < n - 1:
                update = _cpp_gas.update_one(
                    p.omega,
                    p.gamma,
                    p.beta,
                    g_t,
                    obs,
                    copula,
                    result.scaling,
                    score_eps,
                )
                g_t = update.g_next
                r_t = update.r_next
        return samples

    def predict(self, copula, u, result, n, rng=None, **kwargs):
        if rng is None:
            rng = np.random.default_rng()
        r = self.predictive_params(copula, u, result, n, rng=rng, **kwargs)
        d = copula_dimension(copula, u)
        return sample_predictive(
            copula,
            n,
            r,
            given=kwargs.get("given"),
            rng=rng,
            d=d,
        )

    def predictive_params(self, copula, u, result, n, rng=None, **kwargs):
        state = self.predictive_state(copula, u, result, **kwargs)
        return self.sample_params(copula, state, n, rng=rng, **kwargs)

    def predictive_state(self, copula, u, result, **kwargs):
        if u is None or len(u) == 0:
            r_t = float(result.r_last)
        else:
            p = result.params
            r_t = gas_predict_param(
                p.omega,
                p.gamma,
                p.beta,
                u,
                copula,
                result.scaling,
                self._score_eps(result),
                horizon=kwargs.get("horizon", "next"),
            )
        return PredictiveState(
            method="GAS",
            horizon=str(kwargs.get("horizon", "next")).lower(),
            kind="point",
            r=np.array([r_t], dtype=np.float64),
            metadata={
                "g": float(copula.inv_transform(np.array([r_t]))[0])
            }
            if hasattr(copula, "inv_transform")
            else {},
        )

    def condition_state(self, copula, state, observation, result, **kwargs):
        if observation is None:
            return state
        u = np.asarray(observation, dtype=np.float64)
        d = copula_dimension(copula, u)
        if u.ndim != 2 or d is None or u.shape[1] != d or len(u) == 0:
            return state
        u = u[:1]

        p = result.params
        if "g" in state.metadata:
            g_t = float(state.metadata["g"])
        else:
            g_t = float(
                copula.inv_transform(np.array([float(state.r[0])]))[0])
        update = _cpp_gas.update_one(
            p.omega,
            p.gamma,
            p.beta,
            g_t,
            u,
            copula,
            result.scaling,
            self._score_eps(result),
        )
        return PredictiveState(
            method=state.method,
            horizon=state.horizon,
            kind=state.kind,
            r=np.array([update.r_next], dtype=np.float64),
            metadata={**dict(state.metadata), "g": update.g_next},
        )

    def sample_params(self, copula, state, n, rng=None, **kwargs):
        return np.full(n, float(np.asarray(state.r)[0]), dtype=np.float64)

    def model_sample_params(self, copula, result, n, rng=None, **kwargs):
        raise ValueError(
            "GAS sample paths require stepwise score updates and cannot be "
            "precomputed"
        )

    def model_sample_state(self, copula, result, **kwargs):
        p = result.params
        initial = _cpp_gas.initial_state(
            p.omega,
            p.gamma,
            p.beta,
            copula,
            result.scaling,
            self._score_eps(result),
        )
        return PredictiveState(
            method="GAS",
            horizon="model",
            kind="point",
            r=np.array([initial.parameter], dtype=np.float64),
            metadata={"g": initial.g},
        )
