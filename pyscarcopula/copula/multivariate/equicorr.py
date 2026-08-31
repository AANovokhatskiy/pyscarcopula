"""Equicorrelation Gaussian copula."""

import numpy as np
from pyscarcopula.numerical._arrays import (
    as_float64_array,
    as_float64_scalar,
    validate_sampling_n_threads as _validated_n_threads,
)

from pyscarcopula._types import DEFAULT_CONFIG, NumericalConfig
from pyscarcopula._native import model_policy
from pyscarcopula.copula.multivariate.base import (
    MultivariateCopula,
    model_state_locked,
)
from pyscarcopula.copula.multivariate.conditional import (
    fill_given,
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

    def __init__(self, d, rotate=0):
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}")
        super().__init__(
            dimension=d, name=f"Equicorr Gaussian copula (d={d})")
        self._d = d
        self._bounds = model_policy.public_bounds(self)

    @property
    def d(self):
        return self._d

    def transform(self, x):
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.transform(self, x)

    def inv_transform(self, r):
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.inverse_transform(self, r)

    def dtransform(self, x):
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.dtransform(self, x)

    def prepare_sufficient_statistics(
            self,
            u_batches,
            *,
            dimension=None,
            batch_rows=256,
            dimension_tile=16384,
            n_threads=1):
        """Prepare O(T) emission statistics from dense or streamed blocks.

        ``u_batches`` may be a 2D ndarray/memmap or a repeatable or one-shot
        iterable yielding 2D blocks of pseudo-observations. The default is
        unconditionally single-threaded.
        """
        from pyscarcopula.copula.multivariate.equicorr_prepared import (
            EquicorrPreparedData,
        )
        from pyscarcopula._native import multivariate as multivariate_native

        def positive_integer(name, value):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer")
            value = int(value)
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
            return value

        expected_d = self._d if dimension is None else positive_integer(
            "dimension", dimension)
        if expected_d != self._d:
            raise ValueError(
                f"dimension must equal model dimension {self._d}, "
                f"got {expected_d}")
        batch_rows = positive_integer("batch_rows", batch_rows)
        dimension_tile = positive_integer(
            "dimension_tile", dimension_tile)
        n_threads = multivariate_native._validated_n_threads(n_threads)

        arrays = []
        arrays2 = []
        clipping_events = 0
        nonfinite_values = 0
        preparation_blocks = 0
        max_parallel_blocks = 0
        parallel_axes = set()
        peak_temporary_values = 0

        if isinstance(u_batches, np.ndarray):
            sources = (u_batches,)
        else:
            try:
                sources = iter(u_batches)
            except TypeError as error:
                raise ValueError(
                    "u_batches must be a 2D array or an iterable of "
                    "2D blocks") from error

        for source in sources:
            block = as_float64_array(source, name="u_batches block")
            if block.ndim != 2 or block.shape[1] != expected_d:
                raise ValueError(
                    f"each block must have shape (T, {expected_d}), "
                    f"got {block.shape}")
            for begin in range(0, len(block), batch_rows):
                batch = block[begin:begin + batch_rows]
                if len(batch) == 0:
                    continue
                sum_z, sum_z2, diagnostics = (
                    multivariate_native.prepare_equicorr_statistics(
                        batch,
                        dimension_tile=dimension_tile,
                        n_threads=n_threads,
                    )
                )
                arrays.append(sum_z)
                arrays2.append(sum_z2)
                preparation_blocks += 1
                clipping_events += diagnostics["clipping_events"]
                nonfinite_values += diagnostics["nonfinite_values"]
                max_parallel_blocks = max(
                    max_parallel_blocks, diagnostics["parallel_blocks"])
                parallel_axes.add(diagnostics["parallel_axis"])
                peak_temporary_values = max(
                    peak_temporary_values,
                    diagnostics["temporary_values"])
        if not arrays:
            raise ValueError("u_batches must contain at least one observation")

        sum_z = np.concatenate(arrays)
        sum_z2 = np.concatenate(arrays2)
        return EquicorrPreparedData(
            sum_z=sum_z,
            sum_z2=sum_z2,
            n_obs=len(sum_z),
            dimension=expected_d,
            diagnostics={
                "preparation_blocks": preparation_blocks,
                "batch_rows": batch_rows,
                "dimension_tile": dimension_tile,
                "n_threads": n_threads,
                "max_parallel_blocks": max_parallel_blocks,
                "parallel_axes": sorted(parallel_axes),
                "peak_temporary_values": peak_temporary_values,
                "clipping_events": clipping_events,
                "nonfinite_values": nonfinite_values,
            },
            _copy_arrays=False,
        )

    @model_state_locked
    def log_likelihood(self, u, r=None, *, n_threads=1):
        """Evaluate the fitted strategy, or a static density at explicit r."""
        if r is None:
            return self._fitted_log_likelihood(u, n_threads=n_threads)
        r = as_float64_scalar(r, name="r")
        from pyscarcopula._native import static as static_likelihood
        return static_likelihood.prepare(
            self, u, n_threads=n_threads).log_likelihood(r)

    def log_pdf_rows(self, u, r, t_index=None, *, n_threads=1):
        from pyscarcopula._native import multivariate as multivariate_native
        values, _ = multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index, n_threads=n_threads)
        return values

    def dlog_pdf_dr_rows(self, u, r, t_index=None, *, n_threads=1):
        from pyscarcopula._native import multivariate as multivariate_native
        _, values = multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index, n_threads=n_threads)
        return values

    def log_pdf_and_dlog_dr_rows(
            self, u, r, t_index=None, *, n_threads=1):
        from pyscarcopula._native import multivariate as multivariate_native
        return multivariate_native.log_pdf_and_dlog_rows(
            self, u, r, t_index=t_index, n_threads=n_threads)

    def pdf_on_grid(self, u_row, z_grid, *, n_threads=1):
        values, _ = self.pdf_and_grad_on_grid_batch(
            np.asarray(u_row, dtype=np.float64)[None, :],
            z_grid,
            n_threads=n_threads,
        )
        return values[0]

    def pdf_and_grad_on_grid(self, u_row, z_grid, *, n_threads=1):
        values, gradients = self.pdf_and_grad_on_grid_batch(
            np.asarray(u_row, dtype=np.float64)[None, :],
            z_grid,
            n_threads=n_threads,
        )
        return values[0], gradients[0]

    @staticmethod
    def _grid_output_bytes(n_obs, n_grid):
        n_obs = int(n_obs)
        n_grid = int(n_grid)
        if n_obs < 0 or n_grid < 0:
            raise ValueError("grid output dimensions must be non-negative")
        return 2 * n_obs * n_grid * np.dtype(np.float64).itemsize

    def _sample_output_bytes(self, n_rows):
        n_rows = int(n_rows)
        if n_rows < 0:
            raise ValueError("sample output rows must be non-negative")
        return n_rows * self._d * np.dtype(np.float64).itemsize

    @staticmethod
    def _validated_memory_budget(memory_budget_bytes, required, guidance):
        if memory_budget_bytes is None:
            return
        if (
                isinstance(memory_budget_bytes, (bool, np.bool_))
                or not isinstance(
                    memory_budget_bytes, (int, np.integer))):
            raise TypeError("memory_budget_bytes must be an integer")
        if int(memory_budget_bytes) < required:
            raise MemoryError(
                f"output requires {required} bytes; {guidance}")

    def pdf_and_grad_on_grid_batch(
            self,
            u,
            x_grid,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        """Evaluate a grid batch, optionally enforcing an output budget."""
        from pyscarcopula._native import multivariate as multivariate_native
        required = self._grid_output_bytes(len(u), len(x_grid))
        self._validated_memory_budget(
            memory_budget_bytes,
            required,
            "use pdf_and_grad_on_grid_batches() or increase "
            "memory_budget_bytes",
        )
        return multivariate_native.pdf_and_grad_grid(
            self, u, x_grid, n_threads=n_threads)

    def pdf_and_grad_on_grid_batches(
            self,
            u,
            x_grid,
            *,
            batch_rows=128,
            n_threads=1,
            memory_budget_bytes=None):
        """Yield bounded row blocks of the two ``(T,K)`` grid outputs."""
        from pyscarcopula.copula.multivariate.equicorr_prepared import (
            EquicorrPreparedData,
        )

        if isinstance(batch_rows, (bool, np.bool_)) or not isinstance(
                batch_rows, (int, np.integer)):
            raise TypeError("batch_rows must be an integer")
        batch_rows = int(batch_rows)
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        n_grid = len(x_grid)
        per_block = self._grid_output_bytes(
            min(batch_rows, len(u)), n_grid)
        self._validated_memory_budget(
            memory_budget_bytes,
            per_block,
            "reduce batch_rows or increase memory_budget_bytes",
        )

        for start in range(0, len(u), batch_rows):
            stop = min(len(u), start + batch_rows)
            if isinstance(u, EquicorrPreparedData):
                block = EquicorrPreparedData(
                    sum_z=u.sum_z[start:stop],
                    sum_z2=u.sum_z2[start:stop],
                    n_obs=stop - start,
                    dimension=u.dimension,
                    format_version=u.format_version,
                    clipping_epsilon=u.clipping_epsilon,
                    diagnostics={
                        **dict(u.diagnostics),
                        "source_row_range": [start, stop],
                    },
                    _copy_arrays=False,
                )
            else:
                block = u[start:stop]
            yield self.pdf_and_grad_on_grid_batch(
                block,
                x_grid,
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes,
            )

    def copula_grid_batch(
            self, u, x_grid, *, n_threads=1, memory_budget_bytes=None):
        values, _ = self.pdf_and_grad_on_grid_batch(
            u,
            x_grid,
            n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes,
        )
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
        from pyscarcopula.copula.multivariate.equicorr_prepared import (
            EquicorrPreparedData,
        )

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
        from pyscarcopula._native import static as static_likelihood
        evaluator = static_likelihood.prepare(
            self, u, n_threads=config.n_threads)

        from pyscarcopula.strategy.multivariate_mle import (
            StaticMLEEvaluation,
            StaticMLEProblem,
            run_static_multivariate_mle,
        )

        def evaluate(x):
            value, gradient = evaluator.transformed_objective_and_gradient(
                x[0], fail_value=config.fail_value)
            return StaticMLEEvaluation(value, gradient)

        initial, fit_bounds = model_policy.equicorr_fit_policy()
        outcome = run_static_multivariate_mle(
            StaticMLEProblem(
                family="equicorr_gaussian",
                initial_parameters=np.array([initial]),
                bounds=[fit_bounds], evaluate=evaluate),
            optimizer_options=optimizer_options,
            fail_value=config.fail_value,
        )
        rho_hat = self.transform(outcome.parameters)[0]
        fitted = MultivariateMLEResult(
            log_likelihood=-outcome.final_objective,
            method="MLE",
            copula_name=self._name,
            success=outcome.accepted,
            nfev=outcome.nfev,
            message=outcome.message,
            copula_param=rho_hat,
            parameter_count=1,
            n_observations=len(u),
            model_parameters={"rho": rho_hat},
            correlation_matrix=None,
            diagnostics={
                **outcome.diagnostics(),
                "n_threads": config.n_threads,
                "model_score": "not_applicable",
                "optimizer_gradient": "analytical",
                "gradient_kind": "analytical_chain_rule",
                "setup_derivative": "not_applicable",
                "filter_derivative": "not_applicable",
                "parameter_gradient": "analytical_rho",
                "transform_chain_rule": True,
                "corr_matrix": None,
                "correlation_representation": "equicorrelation_scalar",
                "equicorrelation_rho": float(rho_hat),
            },
        )
        if outcome.accepted:
            self.fit_result = fitted
        return fitted

    @model_state_locked
    def fit(
            self,
            data,
            method="scar-tm-ou",
            to_pobs=False,
            config=None,
            **kwargs):
        from pyscarcopula._utils import pobs
        from pyscarcopula.copula.multivariate.equicorr_prepared import (
            EquicorrPreparedData,
        )
        from pyscarcopula.strategy._base import (
            partition_strategy_fit_kwargs,
        )

        partition_strategy_fit_kwargs(method, kwargs)
        if isinstance(data, EquicorrPreparedData):
            if data.dimension != self._d:
                raise ValueError(
                    "prepared dimension does not match model dimension")
            if to_pobs:
                raise ValueError(
                    "to_pobs=True is unavailable for prepared statistics")
            if method.upper() not in {"MLE", "GAS", "SCAR-TM-OU"}:
                raise ValueError(
                    "prepared statistics currently support MLE, GAS, and "
                    "SCAR-TM-OU")
            observations = data
        else:
            observations = as_float64_array(data, name="data")
            if observations.ndim != 2 or observations.shape[1] != self._d:
                raise ValueError(
                    f"data must have shape (n_observations, {self._d})")
            if observations.shape[0] == 0:
                raise ValueError(
                    "data must contain at least one observation")
            if not np.all(np.isfinite(observations)):
                raise ValueError("data must contain only finite values")
            if to_pobs:
                observations = pobs(observations)
                if not np.all(np.isfinite(observations)):
                    raise ValueError(
                        "pseudo-observations must contain only finite values")

        if method.upper() == "MLE":
            optimizer_kwargs = {
                key: kwargs.pop(key)
                for key in _LBFGSB_FIT_KEYS
                if key in kwargs
            }
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(
                    f"unexpected MLE keyword argument(s): {unexpected}")
            result = self._fit_mle(
                observations, config=config, **optimizer_kwargs)
            if not result.success:
                return result
        else:
            from pyscarcopula.api import fit
            return fit(
                self, observations, method=method, config=config, **kwargs)
        if isinstance(observations, EquicorrPreparedData):
            self._last_prepared = observations
            self._last_u = None
        else:
            self._last_u = observations.copy()
            self._last_prepared = None
        return result

    def sample_at_parameter(
            self, n, r, rng=None, *, n_threads=1, memory_budget_bytes=None):
        n_threads = _validated_n_threads(n_threads)
        if isinstance(n, (bool, np.bool_)) or not isinstance(
                n, (int, np.integer)):
            raise TypeError("n must be an integer")
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        self._validated_memory_budget(
            memory_budget_bytes,
            self._sample_output_bytes(n),
            "use sample_at_parameter_batches() or increase "
            "memory_budget_bytes",
        )
        if rng is None:
            rng = np.random.default_rng()
        parameters = np.atleast_1d(as_float64_array(r, name="r")).ravel()
        if parameters.size == 1:
            parameters = np.full(n, parameters[0], dtype=np.float64)
        elif parameters.size != n:
            raise ValueError(
                f"r must be scalar or array of length {n}, "
                f"got {parameters.size}")
        from pyscarcopula._native import multivariate as multivariate_native
        multivariate_native.validate_equicorrelation_path(
            parameters, self._d, n, name="r")
        normal = rng.standard_normal((n, self._d))
        common_count = multivariate_native.equicorr_gaussian_common_draw_count(
            parameters, self._d, n)
        common = rng.standard_normal(common_count)
        return multivariate_native.equicorr_gaussian_sample_from_normals(
            parameters,
            self._d,
            normal,
            common,
            n_threads=n_threads,
        )

    def sample_at_parameter_batches(
            self,
            n,
            r,
            *,
            batch_rows=128,
            rng=None,
            n_threads=1,
            memory_budget_bytes=None):
        """Yield unconditional samples without allocating the full ``(n,d)``.

        The returned iterator owns no model state. Each yielded array has at
        most ``batch_rows`` rows and uses the structural equicorrelation
        sampler for both positive and negative correlation.
        """
        n_threads = _validated_n_threads(n_threads)
        if isinstance(n, (bool, np.bool_)) or not isinstance(
                n, (int, np.integer)):
            raise TypeError("n must be an integer")
        if isinstance(batch_rows, (bool, np.bool_)) or not isinstance(
                batch_rows, (int, np.integer)):
            raise TypeError("batch_rows must be an integer")
        n = int(n)
        batch_rows = int(batch_rows)
        if n < 0:
            raise ValueError("n must be non-negative")
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        self._validated_memory_budget(
            memory_budget_bytes,
            self._sample_output_bytes(min(n, batch_rows)),
            "reduce batch_rows or increase memory_budget_bytes",
        )
        if rng is None:
            rng = np.random.default_rng()

        parameters = np.atleast_1d(
            as_float64_array(r, name="r")).ravel()
        if parameters.size not in (1, n):
            raise ValueError(
                f"r must be scalar or array of length {n}, "
                f"got {parameters.size}")
        from pyscarcopula._native import multivariate as multivariate_native
        multivariate_native.validate_equicorrelation_path(
            parameters, self._d, n, name="r")

        for start in range(0, n, batch_rows):
            stop = min(n, start + batch_rows)
            block_r = parameters if parameters.size == 1 else (
                parameters[start:stop])
            yield self.sample_at_parameter(
                stop - start,
                block_r,
                rng=rng,
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes,
            )

    @model_state_locked
    def sample(
            self, n, u=None, rng=None, *, n_threads=1,
            memory_budget_bytes=None):
        """Generate observations reproducing the fitted model."""
        n_threads = _validated_n_threads(n_threads)
        if self.fit_result is None:
            raise ValueError("Fit first")
        self._validated_memory_budget(
            memory_budget_bytes,
            self._sample_output_bytes(n),
            "use sample_batches() or increase memory_budget_bytes",
        )
        from pyscarcopula.api import sample as _api_sample

        u_data = u if u is not None else getattr(self, "_last_u", None)
        if u_data is None:
            u_data = getattr(self, "_last_prepared", None)
        if u_data is None:
            raise ValueError(
                "No data for sample. "
                "Either call fit() first or pass u= explicitly.")
        return _api_sample(
            self, u_data, self.fit_result, n, rng=rng,
            n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes)

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
        """Yield fitted-model samples in bounded row blocks.

        GAS is advanced one generated observation at a time. MLE and SCAR
        use their constant or OU model parameter paths respectively.
        """
        n_threads = _validated_n_threads(n_threads)
        if self.fit_result is None:
            raise ValueError("Fit first")
        if isinstance(n, (bool, np.bool_)) or not isinstance(
                n, (int, np.integer)):
            raise TypeError("n must be an integer")
        if isinstance(batch_rows, (bool, np.bool_)) or not isinstance(
                batch_rows, (int, np.integer)):
            raise TypeError("batch_rows must be an integer")
        n = int(n)
        batch_rows = int(batch_rows)
        if n < 0:
            raise ValueError("n must be non-negative")
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        self._validated_memory_budget(
            memory_budget_bytes,
            self._sample_output_bytes(min(n, batch_rows)),
            "reduce batch_rows or increase memory_budget_bytes",
        )
        if rng is None:
            rng = np.random.default_rng()

        from pyscarcopula.strategy._base import get_strategy_for_result
        result = self.fit_result
        strategy = get_strategy_for_result(result)
        state = strategy.model_sample_state(self, result)
        from pyscarcopula.strategy.predict_helpers import sample_model_batches
        return sample_model_batches(
            self, strategy, result, state, n, batch_rows=batch_rows,
            given=given, rng=rng, n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes)

    @model_state_locked
    def sample_conditional(
            self,
            n,
            r=None,
            given=None,
            rng=None,
            *,
            n_threads=1,
            memory_budget_bytes=None):
        n_threads = _validated_n_threads(n_threads)
        self._validated_memory_budget(
            memory_budget_bytes,
            self._sample_output_bytes(n),
            "use sample_batches()/predict_batches() or increase "
            "memory_budget_bytes",
        )
        if rng is None:
            rng = np.random.default_rng()
        given = validate_multivariate_given(given, self._d)
        if not given:
            if r is None:
                return self.sample(
                    n, rng=rng, n_threads=n_threads,
                    memory_budget_bytes=memory_budget_bytes)
            return self.sample_at_parameter(
                n,
                r=r,
                rng=rng,
                n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes,
            )
        if r is None and len(given) == self._d:
            return fill_given(n, self._d, given)
        if r is None:
            return self.predict(
                n, given=given, rng=rng, n_threads=n_threads,
                memory_budget_bytes=memory_budget_bytes)
        return sample_gaussian_conditional(
            n, self._d, r, given=given, rng=rng,
            n_threads=n_threads)

    @model_state_locked
    def predict(
            self,
            n,
            u=None,
            rng=None,
            given=None,
            horizon="next",
            predictive_r_mode=None,
            predict_config=None,
            memory_budget_bytes=None,
            *,
            n_threads=1):
        n_threads = _validated_n_threads(n_threads)
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
        self._validated_memory_budget(
            memory_budget_bytes,
            self._sample_output_bytes(n),
            "use predict_batches() or increase memory_budget_bytes",
        )
        if rng is None:
            rng = np.random.default_rng()

        from pyscarcopula._types import MLEResult
        if isinstance(self.fit_result, MLEResult):
            return self.sample_conditional(
                n, r=self.fit_result.copula_param, given=given, rng=rng,
                n_threads=n_threads, memory_budget_bytes=memory_budget_bytes)

        observations = u if u is not None else getattr(self, "_last_u", None)
        if observations is None:
            observations = getattr(self, "_last_prepared", None)
        blocks = self.predict_batches(
            n,
            u=observations,
            rng=rng,
            batch_rows=max(1, n),
            given=given,
            horizon=horizon,
            predictive_r_mode=predictive_r_mode,
            n_threads=n_threads,
            memory_budget_bytes=memory_budget_bytes,
        )
        try:
            return next(blocks)
        except StopIteration:
            return np.empty((0, self._d), dtype=np.float64)

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
        """Yield fitted predictive samples from one frozen predictive state."""
        n_threads = _validated_n_threads(n_threads)
        if self.fit_result is None:
            raise ValueError("Fit first")
        if isinstance(n, (bool, np.bool_)) or not isinstance(
                n, (int, np.integer)):
            raise TypeError("n must be an integer")
        if isinstance(batch_rows, (bool, np.bool_)) or not isinstance(
                batch_rows, (int, np.integer)):
            raise TypeError("batch_rows must be an integer")
        n = int(n)
        batch_rows = int(batch_rows)
        if n < 0:
            raise ValueError("n must be non-negative")
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        self._validated_memory_budget(
            memory_budget_bytes,
            self._sample_output_bytes(min(n, batch_rows)),
            "reduce batch_rows or increase memory_budget_bytes",
        )
        if rng is None:
            rng = np.random.default_rng()

        from pyscarcopula.api import _resolve_predict_config
        config = _resolve_predict_config(
            predict_config,
            given,
            horizon,
            {"predictive_r_mode": predictive_r_mode},
        )
        observations = u if u is not None else getattr(
            self, "_last_u", None)
        if observations is None:
            observations = getattr(self, "_last_prepared", None)

        from pyscarcopula.strategy._base import get_strategy_for_result
        result = self.fit_result
        strategy = get_strategy_for_result(result)
        state = strategy.predictive_state(
            self,
            observations,
            result,
            horizon=config.horizon,
            predictive_r_mode=config.predictive_r_mode,
        )

        from pyscarcopula.strategy.predict_helpers import sample_predictive_batches
        return sample_predictive_batches(
            self, strategy, state, n, batch_rows=batch_rows,
            given=config.given, rng=rng,
            predictive_r_mode=config.predictive_r_mode,
            n_threads=n_threads, memory_budget_bytes=memory_budget_bytes)

    @model_state_locked
    def predictive_mean(self, u):
        """Return the predictive parameter path for the fitted strategy."""
        if self.fit_result is None:
            raise ValueError("Fit the model before calling predictive_mean")
        from pyscarcopula.api import predictive_mean as _predictive_mean

        return _predictive_mean(self, u, self.fit_result)

    @model_state_locked
    def xT_distribution(self, u, K=300, grid_range=5.0):
        if self.fit_result is None:
            raise ValueError("Fit with SCAR first")
        kappa, mu, nu = self.fit_result.params.values
        from pyscarcopula._native import scar_ou as _cpp_scar_ou
        from pyscarcopula.numerical._scar_ou_config import AutoTMConfig
        from pyscarcopula.copula.multivariate.equicorr_prepared import (
            EquicorrPreparedData,
        )
        config = AutoTMConfig(K=K, grid_range=grid_range)
        if isinstance(u, EquicorrPreparedData):
            return _cpp_scar_ou.prepare_objective(
                u, self, config).state_distribution(
                    kappa, mu, nu, horizon="current")
        return _cpp_scar_ou.state_distribution(
            kappa,
            mu,
            nu,
            u,
            self,
            config,
        )
