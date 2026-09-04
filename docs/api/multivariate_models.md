# Multivariate Models API

Choose a focused reference below. The [model guide](../guide/multivariate_models.md)
explains correlation modes and model selection. [Configuration and Results](configuration.md)
documents optimizer settings, `MultivariateMLEResult`, and dynamic fit results.

<a id="static-gaussian-and-student-copulas"></a>
<a id="gaussiancopula"></a>
<a id="pyscarcopula.copula.multivariate.gaussian.GaussianCopula"></a>
<a id="studentcopula"></a>
<a id="pyscarcopula.copula.multivariate.student.StudentCopula"></a>
<a id="gaussian-factor-correlation"></a>
<a id="static-student-factor-correlation"></a>


## Static Gaussian and Student Copulas reference

Continue to [Static Gaussian and Student Copulas](static-models.md).

<a id="factor-correlation-operator"></a>
<a id="api"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.dimension"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.rank"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.storage_bytes"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.uniqueness"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.from_unconstrained"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.load_mmap"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.load_npz"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.prepare"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.save_mmap"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.save_npz"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.FactorCorrelation.to_dense"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.dimension"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.loadings"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.logdet"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.rank"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.uniqueness"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.matvec"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.quadratic_form"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.quadratic_forms"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.sample_normal"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.sample_normal_batches"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.solve"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.to_dense"></a>
<a id="pyscarcopula.copula.multivariate.factor_correlation.PreparedFactorCorrelation.transform_normal_draws"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.correlation"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.dimension"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.n_observations"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.observations"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.rank"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.dlog_pdf_ddf_rows"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.evaluate"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.evaluate_grid"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.evaluate_grid_batches"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.joint_likelihood_and_gradient"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.log_likelihood_and_gradient"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.log_pdf_and_dlog_ddf_grid"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.log_pdf_and_dlog_ddf_rows"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.log_pdf_rows"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.objective_and_gradient"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.pdf_and_grad_on_grid"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.pdf_and_grad_on_grid_batches"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.penalized_parameterized_objective_and_gradient"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluator.stochastic_pdf_and_gradient_grid"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluation"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluation.dlog_likelihood_ddf"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluation.dnegative_log_likelihood_ddf"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluation.log_likelihood"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentEvaluation.negative_log_likelihood"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentGridEvaluation"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentGridEvaluation.pdf_and_gradient"></a>
<a id="pyscarcopula.copula.multivariate.factor_student.FactorStudentJointEvaluation"></a>


## Factor correlation operator reference

Continue to [Factor correlation operator](factor.md).

<a id="equicorrelation-gaussian-copula"></a>
<a id="usage"></a>
<a id="high-dimensional-preparation"></a>
<a id="goodness-of-fit"></a>
<a id="sampling"></a>
<a id="api_1"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.fit"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.prepare_sufficient_statistics"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.sample"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.sample_batches"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.sample_conditional"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.predict"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.predict_batches"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.predictive_mean"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.xT_distribution"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.log_likelihood"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.log_pdf_rows"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.dlog_pdf_dr_rows"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.log_pdf_and_dlog_dr_rows"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.pdf_on_grid"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.pdf_and_grad_on_grid"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.pdf_and_grad_on_grid_batch"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.pdf_and_grad_on_grid_batches"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.sample_at_parameter"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.sample_at_parameter_batches"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.transform"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.inv_transform"></a>
<a id="pyscarcopula.copula.multivariate.equicorr.EquicorrGaussianCopula.dtransform"></a>


## Equicorrelation Gaussian Copula reference

Continue to [Equicorrelation Gaussian Copula](equicorrelation.md).

<a id="stochasticstudentcopula"></a>
<a id="stochastic-student-copula-with-estimated-static-correlation"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.fit"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.sample_at_parameter"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.sample_at_parameter_batches"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.sample"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.sample_batches"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.sample_conditional"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.predict"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.predict_batches"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.predictive_mean"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.xT_distribution"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.log_likelihood"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.log_pdf_rows"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.log_pdf_and_dlog_dr_rows"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.pdf_on_grid"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.pdf_and_grad_on_grid"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.transform"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.inv_transform"></a>
<a id="pyscarcopula.copula.multivariate.stochastic_student.StochasticStudentCopula.dtransform"></a>


## StochasticStudentCopula reference

Continue to [StochasticStudentCopula](stochastic-student.md).

<a id="pyscarcopula.MultivariateMLEResult"></a>
<a id="pyscarcopula.MultivariateMLEResult.aic"></a>
<a id="pyscarcopula.MultivariateMLEResult.bic"></a>


## Fit result fields reference

Continue to [Fit result fields](configuration.md).
