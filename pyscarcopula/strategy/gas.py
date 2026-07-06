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
from pyscarcopula.strategy.predict_helpers import (
    predict_from_strategy,
    sample_predictive,
)


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

    def _ensure_correlation_initialized(self, copula, u):
        ensure = getattr(copula, "_ensure_corr_initialized", None)
        if callable(ensure):
            ensure(u)

    def _correlation_diagnostics(self, copula) -> dict:
        diagnostics = {}
        count_diagnostics = getattr(copula, "_corr_count_diagnostics", None)
        if callable(count_diagnostics):
            diagnostics.update(count_diagnostics())
        preprocessing_diagnostics = getattr(
            copula, "correlation_preprocessing_diagnostics", None)
        if callable(preprocessing_diagnostics):
            diagnostics.update(preprocessing_diagnostics())
        corr_params = getattr(copula, "corr_params", None)
        if callable(corr_params):
            diagnostics["corr_params_raw"] = corr_params()
        corr_alpha = getattr(copula, "corr_alpha", None)
        if callable(corr_alpha):
            diagnostics["corr_alpha"] = corr_alpha()
        R = getattr(copula, "R", None)
        if R is not None:
            diagnostics["corr_matrix"] = R
        return diagnostics

    def _build_result(
        self,
        copula,
        u,
        result,
        gas_values,
        score_eps,
        gamma_bound,
        beta_bound,
        *,
        parameter_count=None,
        diagnostics=None,
    ):
        gas_values = np.asarray(gas_values, dtype=np.float64).reshape(-1)
        params = gas_params(
            omega=gas_values[0],
            gamma=gas_values[1],
            beta=gas_values[2],
            gamma_bound=gamma_bound,
            beta_bound=beta_bound,
        )

        success = bool(result.success)
        message = str(result.message)
        try:
            final_log_likelihood = gas_loglik(
                gas_values[0],
                gas_values[1],
                gas_values[2],
                u,
                copula,
                self.scaling,
                score_eps,
            )
            if not np.isfinite(final_log_likelihood):
                raise FloatingPointError(
                    "final GAS log-likelihood is not finite")
            r_last = gas_predict_param(
                gas_values[0],
                gas_values[1],
                gas_values[2],
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

        result_diagnostics = {
            "model_score": "native",
            "optimizer_gradient": "numerical",
            "gradient_kind": "numerical_optimizer",
            "setup_derivative": "not_provided",
            "filter_derivative": "not_provided_to_optimizer",
            "analytical_grad_requested": False,
            "analytical_grad_used": False,
        }
        result_diagnostics.update(self._correlation_diagnostics(copula))
        if diagnostics:
            result_diagnostics.update(diagnostics)

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
            diagnostics=result_diagnostics,
            parameter_count=parameter_count,
        )

    def _fit_joint_static_shrinkage(
        self,
        copula,
        u,
        gamma0,
        optimizer_options,
        score_eps,
        gamma_bound,
        beta_bound,
        verbose,
    ):
        n_corr = int(copula._corr_num_params())
        self._ensure_correlation_initialized(copula, u)
        corr0 = np.asarray(
            copula._initial_corr_params(u), dtype=np.float64).reshape(-1)
        if n_corr != 1 or corr0.size != 1:
            raise NotImplementedError(
                "GAS joint static correlation currently supports only "
                "corr_mode='shrinkage'")

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
            gas0 = np.array([mu_mle * 0.05, 0.05, 0.95])
            fitted_corr = np.asarray(
                copula._pack_corr_params(), dtype=np.float64).reshape(-1)
            if fitted_corr.size == n_corr:
                corr0 = fitted_corr
        else:
            gamma0 = np.asarray(gamma0, dtype=np.float64).reshape(-1)
            if gamma0.size == 3:
                gas0 = gamma0.copy()
            elif gamma0.size == 3 + n_corr:
                gas0 = gamma0[:3].copy()
                corr0 = gamma0[3:].copy()
            else:
                raise ValueError(
                    f"gamma0 must contain 3 GAS parameters or "
                    f"{3 + n_corr} joint parameters, got {gamma0.size}")

        joint0 = np.concatenate([gas0, corr0])
        if not np.all(np.isfinite(joint0)):
            raise ValueError("gamma0 must contain only finite values")

        bounds = Bounds(
            [-np.inf, -gamma_bound, -beta_bound, -np.inf],
            [np.inf, gamma_bound, beta_bound, np.inf],
        )

        def objective(joint):
            joint = np.asarray(joint, dtype=np.float64).reshape(-1)
            if joint.size != 3 + n_corr or not np.all(np.isfinite(joint)):
                return self.config.fail_value
            try:
                copula._set_corr_from_params(joint[3:])
                return gas_negloglik(
                    joint[0],
                    joint[1],
                    joint[2],
                    u,
                    copula,
                    self.scaling,
                    score_eps,
                )
            except Exception:
                return self.config.fail_value

        if verbose:
            print(
                f"GAS fit: joint shrinkage gamma0={joint0}, "
                f"scaling={self.scaling}, score_eps={score_eps}, "
                f"options={optimizer_options}, gamma_bound={gamma_bound}, "
                f"beta_bound={beta_bound}"
            )

        result = minimize(
            objective,
            joint0,
            method="L-BFGS-B",
            bounds=bounds,
            options=optimizer_options,
        )
        try:
            copula._set_corr_from_params(result.x[3:])
        except Exception as exc:
            result.success = False
            result.message = (
                f"{result.message}; failed to set final correlation: {exc}")
            copula._set_corr_from_params(corr0)

        diagnostics = {
            "joint_static": True,
            "joint_optimizer": "python-lbfgsb",
            "joint_correlation": "shrinkage",
            "initial_params": joint0.copy(),
            "final_params": np.asarray(result.x, dtype=np.float64).copy(),
        }
        return self._build_result(
            copula,
            u,
            result,
            result.x[:3],
            score_eps,
            gamma_bound,
            beta_bound,
            parameter_count=3 + n_corr,
            diagnostics=diagnostics,
        )

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
        self._ensure_correlation_initialized(copula, u)
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

        corr_num_params = int(
            getattr(copula, "_corr_num_params", lambda: 0)())
        if corr_num_params:
            if getattr(copula, "_corr_mode", None) == "shrinkage":
                return self._fit_joint_static_shrinkage(
                    copula,
                    u,
                    gamma0,
                    optimizer_options,
                    score_eps,
                    gamma_bound,
                    beta_bound,
                    verbose,
                )
            raise NotImplementedError(
                "GAS joint static correlation currently supports only "
                "corr_mode='shrinkage'")

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
        parameter_count = None
        corr_effective_num_params = getattr(
            copula, "_corr_effective_num_params", None)
        if callable(corr_effective_num_params):
            parameter_count = 3 + int(corr_effective_num_params())
        return self._build_result(
            copula,
            u,
            result,
            result.x,
            score_eps,
            gamma_bound,
            beta_bound,
            parameter_count=parameter_count,
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
        given = kwargs.get("given")
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

        for t in range(n):
            obs = sample_predictive(
                copula,
                1,
                np.array([r_t]),
                given=given,
                rng=rng,
                d=d,
            )
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
        return predict_from_strategy(
            self, copula, u, result, n, rng=rng, **kwargs)

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
