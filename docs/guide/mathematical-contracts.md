# Mathematical Contracts

This page gives the compact mathematical contract behind the public fitting,
prediction, and goodness-of-fit APIs. It is intentionally shorter than the
full derivation notes in `mathematical_algorithm_description/`; the goal here
is to explain what the package computes and which numerical approximations are
part of each model.

## Common Notation

The algorithms operate on pseudo-observations

$$
u_t=(u_{t1},\ldots,u_{td}) \in (0,1)^d,\qquad t=1,\ldots,T.
$$

For continuous margins, Sklar's factorization separates marginal modeling
from dependence modeling:

$$
f_t(y_1,\ldots,y_d)
= c_t(u_1,\ldots,u_d)\prod_{j=1}^d f_j(y_j),
\qquad u_j=F_j(y_j).
$$

`pyscarcopula` assumes that the marginal transformation has already produced
the pseudo-observations. Dynamic copulas then model a scalar dependence
parameter

$$
\theta_t = \Psi(x_t)
$$

where `Psi` maps an unconstrained state into the valid copula-parameter
domain. For a latent state value `x`, the observation or emission density is

$$
f_t(x) = c(u_t; \Psi(x)).
$$

The generic state derivative used by dynamic gradients is

$$
\frac{\partial f_t(x)}{\partial x}
=
f_t(x)
\left.
\frac{\partial \log c(u_t;\theta)}{\partial \theta}
\right|_{\theta=\Psi(x)}
\Psi'(x).
$$

This identity is the bridge between analytical copula scores and the GAS and
SCAR filters.

## Parameter Links

The dynamic state is usually unconstrained, while copula parameters are not.
The public models therefore use smooth links:

- positive-parameter families use a shifted softplus link,
  $\Psi(x)=a+\log(1+\exp(x))$;
- bivariate Gaussian dependence uses a bounded tanh link;
- equicorrelation Gaussian dependence uses a dimension-aware bounded link
  into $(-1/(d-1),1)$;
- Student degrees of freedom use
  $\nu_t=2+10^{-6}+\log(1+\exp(x_t))$ so the fitted copula has finite
  variance.

Some bivariate copulas can also use the `xtanh` transform. It is a valid
forward transform for fitting, but its positive-branch inverse is only an
initialization convention because the map is not globally one-to-one.

Pseudo-observations are clipped away from 0 and 1 before Gaussian or Student
quantiles are evaluated. That is a numerical safety operation, not a change in
the copula model.

## MLE

MLE assumes a constant copula parameter:

$$
\ell(\theta)=\sum_{t=1}^T \log c(u_t;\theta).
$$

The optimizer works in the natural copula-parameter space. For example,
`alpha0=[2.0]` for a Gumbel copula means a Gumbel parameter of 2.0, and
`alpha0=[5.0]` for a Student copula means five degrees of freedom. Dynamic
latent transforms are not part of the MLE objective.

## GAS

GAS is an observation-driven model. Conditional on the past, the next copula
parameter is a point value:

$$
\theta_t=\Psi(g_t),
\qquad
g_{t+1}=\omega+\beta g_t+\gamma s_t.
$$

Here `s_t` is the scaled score of the current copula log-density with respect
to the recursion state. In unit scaling,

$$
s_t =
\frac{\partial \log c(u_t;\Psi(g_t))}{\partial g_t}.
$$

Fisher scaling rescales this score by a curvature estimate. It is available
for experimentation, but it combines finite-difference curvature, clipping,
and floors inside the recursion, so `scaling='unit'` is the recommended
production choice.

The native GAS evaluator owns the likelihood, score recursion, filtering,
state updates, prediction state, and bivariate Rosenblatt path for supported
models. The score used in the recursion is not the optimizer Jacobian with
respect to $(\omega,\gamma,\beta)$; the outer L-BFGS-B gradient is currently
numerical.

## SCAR-TM-OU

SCAR-TM-OU is a parameter-driven latent-state model. The unconstrained state
follows an Ornstein-Uhlenbeck diffusion:

$$
dX_t=\kappa(\mu-X_t)\,dt+\nu\,dW_t,
\qquad
\theta_t=\Psi(X_t).
$$

For one observation step, with

$$
\sigma^2=\frac{\nu^2}{2\kappa},
\qquad
\rho=\exp(-\kappa dt),
$$

the exact transition is

$$
X_k \mid X_{k-1}=x
\sim
N\left(\mu+\rho(x-\mu),\sigma^2(1-\rho^2)\right).
$$

The likelihood integrates over the whole latent path:

$$
L =
\int
\prod_{t=1}^T c(u_t;\Psi(x_t))
p(x_t \mid x_{t-1})
\,dx_{1:T}.
$$

Because the latent process is one-dimensional Markov, the package evaluates
this integral by deterministic filtering rather than by Monte Carlo
trajectory averaging.

### OU Backends

`transition_method='spectral'` uses the stationary OU representation. In the
standardized coordinate $X_t=\mu+\sigma Z_t$, the OU transition is diagonal in
the orthonormal Hermite basis:

$$
P_\rho \psi_n = \rho^n \psi_n.
$$

Each observation multiplies by the emission factor
$c(u_t;\Psi(\mu+\sigma z))$ and projects back to the truncated Hermite basis
by Gauss-Hermite quadrature. This is fast when $\kappa dt$ is not too small.

`transition_method='matrix'` discretizes the OU state on a finite grid and
uses a weighted transition matrix

$$
T_{ji} \approx p(x_j \mid x_i) w_j.
$$

The backward recursion has the form

$$
m_{k-1,i}=\sum_j T_{ji} f_k(x_j)m_{k,j}.
$$

`transition_method='local'` avoids a full transition matrix. For each previous
grid point, it applies a local Gauss-Hermite rule to the conditional Gaussian
transition and interpolates off-grid values. This is usually safer when the
one-step OU kernel is very narrow.

`transition_method='auto'` chooses spectral outside the narrow-kernel regime,
uses local for small $\kappa dt$, and treats matrix then local as numerical
fallbacks if spectral evaluation fails.

### OU Gradients

With `analytical_grad=True`, SCAR-TM-OU passes a native analytical Jacobian to
the optimizer. The derivative differentiates both the emission terms and the
normalized filtering recursion. For joint Stochastic Student fits, the native
engine supplies OU and static-correlation derivatives, and Python applies the
configured correlation-parameter chain rule.

## SCAR-TM-JACOBI

SCAR-TM-JACOBI evolves Kendall's tau directly inside `(0, 1)`:

$$
d\tau_t =
\kappa(m-\tau_t)\,dt
+ \xi\sqrt{\tau_t(1-\tau_t)}\,dW_t.
$$

The copula parameter is recovered through the model's `tau_to_param` mapping.
This method is therefore available only for copulas that expose both
`tau_to_param` and `param_to_tau`.

The stationary law is beta with shape parameters

$$
a=\frac{2\kappa m}{\xi^2},
\qquad
b=\frac{2\kappa(1-m)}{\xi^2}.
$$

The spectral backend uses the Jacobi eigenbasis associated with this
stationary law. The matrix backend applies the transition on a tau quadrature
grid. The local backend uses the Lamperti coordinate

$$
y=\frac{2}{\xi}\arcsin\sqrt{\tau}
$$

and maps local Gauss-Hermite nodes back to tau space. For high-frequency data,
`dt = 1 / (T - 1)`, so one-step transitions can be close to a point mass. In
that regime, truncated global Jacobi expansions can create negative entries
or invalid row sums; `transition_method='auto'` therefore falls back to the
local backend when the spectral matrix is not acceptable.

Jacobi gradients are fully analytical for `local_fixed`. For `local`,
`spectral_matrix`, and `auto`, setup-level arrays are differentiated
numerically while the filtering recursion is differentiated analytically; the
reported gradient kind is therefore `semi_analytical`.

## Multivariate Scalar-State Models

The multivariate dynamic models use the same scalar-state strategy contract:
the model supplies row-wise densities and, for GAS, row-wise score
derivatives.

For the equicorrelation Gaussian copula,

$$
R(\rho)=(1-\rho)I+\rho \mathbf{1}\mathbf{1}^\top,
\qquad
\rho \in \left(-\frac{1}{d-1},1\right).
$$

For the Stochastic Student copula,

$$
c(u_t;R,\nu_t)=
\frac{t_d(q_t;0,R,\nu_t)}
     {\prod_{j=1}^d t_1(q_{tj};\nu_t)},
\qquad
q_{tj}=T_{\nu_t}^{-1}(u_{tj}).
$$

The dynamic state controls $\nu_t$ and therefore tail thickness. Static
correlation can be fixed, estimated through one-parameter shrinkage, or
estimated through a Cholesky parameterization. Kendall preprocessing maps
pairwise tau estimates by $R_{ij}=\sin(\pi\tau_{ij}/2)$ and projects to an SPD
correlation matrix when needed.

## Dynamic Rosenblatt GoF

Goodness-of-fit tests evaluate calibration by transforming fitted conditional
observations to variables that should be independent uniforms under the model.
The scalar statistic is the Cramer-von Mises reduction of those transformed
values, calibrated by parametric bootstrap when requested.

The important distinction is the state used by the conditional CDF:

- MLE uses a fixed fitted parameter.
- GAS evaluates the conditional distribution at the filtered point state.
- SCAR integrates the conditional distribution over the predictive latent
  state distribution.

For a bivariate SCAR fit, the second Rosenblatt component has the form

$$
v_{t2}
=
\int h_2(u_{t2}\mid u_{t1};\Psi(x))\,
p(x_t \mid u_{1:t-1})\,dx.
$$

The observation at time `t` is not absorbed before computing this predictive
mixture. This is why SCAR GoF differs from applying a point-parameter
Rosenblatt transform to a posterior mean path.

## Sampling And Prediction

`sample` and `predict` answer different questions.

`sample` reproduces the fitted model. For a stochastic dynamic model, it
simulates a new latent or score-driven path and then samples observations from
the copula along that path.

`predict` conditions on the supplied history. MLE uses the fixed fitted
parameter. GAS uses the last filtered score state or the one-step-ahead score
state, depending on `horizon`. SCAR uses either the posterior latent
distribution after the last observation (`horizon='current'`) or the
one-step-ahead predictive latent distribution (`horizon='next'`).

For conditional prediction, fixed `given` values live in pseudo-observation
space. Conditional sampling changes which components are drawn. Dynamic
conditioning, where supported, is separate: it lets fixed prediction-time
values update strategy-owned dynamic states before downstream samples are
generated.

## Numerical Guidance

There are two different convergence questions:

- optimizer convergence asks whether L-BFGS-B has found a stable optimum for
  the current numerical approximation;
- approximation convergence asks whether the transfer grid, basis order, or
  quadrature rule is accurate enough for the fitted model.

For SCAR-TM-OU, compare `auto`, `spectral`, `matrix`, and `local` at important
fit points when numerical sensitivity matters. For spectral likelihoods,
increase `spectral_basis_order`; for grid likelihoods, increase `K`,
`grid_range`, or `pts_per_sigma`; for local transitions, increase `gh_order`
only after the grid itself is adequate.

For SCAR-TM-JACOBI, check whether `auto` selected `spectral_matrix` or `local`.
Negative spectral mass, invalid row sums, or strong basis-order sensitivity are
signs that the local backend is the more reliable approximation.

The diagnostics fields documented in [Estimation Methods](estimation-methods.md)
and [Diagnostics API](../api/diagnostics.md) expose the selected backends,
gradient kind, fallback counters, optimizer status, and correlation
preprocessing outcomes needed for these checks.
