# Mathematical Contracts

This page gives the compact mathematical contract behind the public fitting,
prediction, and goodness-of-fit APIs. Its goal is to explain what the package
computes and which numerical approximations are part of each model.

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

The public dynamic models use an unconstrained state and map it into the
copula parameter domain with smooth links:

- positive-parameter families use a selectable shifted link. The default is
  softplus, $\Psi(x)=a+\log(1+\exp(x))$; `exp` uses
  $\Psi(x)=a+\exp(x)$; and `logistic` uses
  $\Psi(x)=a+20\,\sigma(x/2)$ with range $(a,a+20)$;
- bivariate Gaussian dependence uses a bounded tanh link;
- equicorrelation Gaussian dependence uses a dimension-aware bounded link
  into $(-1/(d-1),1)$;
- Student degrees of freedom use
  $\nu_t=2+10^{-6}+\log(1+\exp(x_t))$ so the fitted copula has finite
  variance.

Some bivariate copulas can also use the `xtanh` transform. It is a valid
forward transform for fitting, but its positive-branch inverse is only an
initialization convention because the map is not globally one-to-one.
For the `exp` and `logistic` links, inverse transforms reject parameters
outside their mathematical ranges. Exact range endpoints use finite capped
latent values solely as an optimizer-initialization convention.

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

Fisher scaling rescales this score by a curvature estimate. It combines
finite-difference curvature, clipping, and floors inside the recursion.
`scaling='unit'` avoids those nested numerical operations and is the baseline
used by the fitting guide.

The compiled GAS evaluator handles likelihood, score recursion, filtering,
state updates, prediction state, and the bivariate Rosenblatt path for
supported models. The score used in the recursion is not the optimizer
Jacobian with respect to $(\omega,\gamma,\beta)$; the outer L-BFGS-B gradient
is numerical.

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
\int p_0(x_1)c(u_1;\Psi(x_1))
\prod_{t=2}^T
p(x_t \mid x_{t-1})c(u_t;\Psi(x_t))
\,dx_{1:T},
$$

Because the latent process is one-dimensional Markov, the package evaluates
this integral by deterministic filtering rather than by Monte Carlo
trajectory averaging.

Here $p_0$ is the stationary OU density,
$N(\mu,\nu^2/(2\kappa))$.

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
transition and interpolates off-grid values. This avoids representing a
one-step OU kernel narrower than the spacing of a fixed global grid.

`transition_method='auto'` chooses spectral outside the narrow-kernel regime,
uses local for small $\kappa dt$, and treats matrix then local as numerical
fallbacks if spectral evaluation fails.

### OU Gradients

With `analytical_grad=True`, SCAR-TM-OU passes an analytical Jacobian to the
optimizer. The derivative differentiates both the emission terms and the
normalized filtering recursion. For joint Stochastic Student fits, the
compiled engine supplies OU and static-correlation derivatives, and Python
applies the configured correlation-parameter chain rule.

`StochasticStudentCopula` additionally reparameterizes the OU block during
optimization. If $\sigma_x=\nu/\sqrt{2\kappa}$ and
$y=(\log\kappa,\mu,\log\sigma_x)$, the public likelihood remains a function
of $\alpha=(\kappa,\mu,\nu)$, while the optimizer receives

$$
\frac{\partial \ell}{\partial y_1}
=\kappa\frac{\partial\ell}{\partial\kappa}
+\frac{\nu}{2}\frac{\partial\ell}{\partial\nu},\qquad
\frac{\partial \ell}{\partial y_2}
=\frac{\partial\ell}{\partial\mu},\qquad
\frac{\partial \ell}{\partial y_3}
=\nu\frac{\partial\ell}{\partial\nu}.
$$

Inputs and results are converted at the strategy boundary, so `alpha0`,
`LatentResult.params`, likelihood evaluation, serialization, and prediction
continue to use $(\kappa,\mu,\nu)$.

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

The implemented state space covers positive Kendall dependence only. For
families such as Frank and bivariate Gaussian, `scar-tm-jacobi` therefore uses
the positive-dependence part of the family.

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
local backend when the spectral matrix is not acceptable. Negative spectral
mass within `negative_mass_tol` is clipped and row-normalized as numerical
truncation noise. Material negative mass is never passed to the probability
filter: `auto` falls back, while an explicit spectral backend fails unless
clipping was explicitly requested.

Jacobi gradients are fully analytical for `local_fixed`. For `local`,
`spectral_matrix`, and `auto`, setup-level arrays are differentiated
numerically while the filtering recursion is differentiated analytically; the
reported gradient kind is therefore `semi_analytical`. The backend selected at
the central point is held fixed across setup finite differences, and the
ordinary likelihood is independently recomputed at the final optimizer point
before a fit can be reported as successful.

The numerical boundary validates non-empty bivariate observations, finite
physical initialization (`kappa > 0`, `0 < m < 1`, `xi > 0`), and strict
integer quadrature orders. Jacobi workspaces are preflighted before root
construction and matrix allocation. A hard order cap prevents accidental
multi-gigabyte quadratic requests; `memory_budget_bytes` can impose a smaller
application-specific limit.

Unconditional simulation is defined on this same quadrature state space:

$$
I_0 \sim w,\qquad
I_t \mid I_{t-1}=i \sim P_{i,\cdot},\qquad
\tau_t=\tau_{I_t}.
$$

Thus sampled latent states are grid atoms, not jittered continuous values,
and the transition used for likelihood and simulation has the same
probability contract. With `dt=1/(n-1)`, the spectral first moment satisfies

$$
\mathbb{E}[\tau_{t+1}\mid\tau_t]
=m+(\tau_t-m)e^{-\kappa\,dt}.
$$

The coefficient-only legacy representation uses the probability-safe `auto`
matrix for unconditional sampling because coefficient recursion does not
define categorical transition rows.

For sparse local transitions, an experimental MH correction replaces
off-diagonal proposal mass by

$$
P_{ij}=q_{ij}\min\left(
1,\frac{w_jq_{ji}}{w_iq_{ij}}
\right),\qquad i\ne j,
$$

and puts rejected mass on the diagonal. This satisfies detailed balance with
the discrete stationary weights but may distort conditional moments.

The experimental IPFP alternative balances the stationary joint flux
$Q_{ij}=w_iq_{ij}$ on its existing sparse support until both marginals equal
$w$. No new edges are introduced. Therefore the operation fails explicitly
when the original support cannot represent both stationary marginals; the
implementation does not conceal infeasibility by adding artificial diagonal
mass. Either correction, when selected, is shared by likelihood, filtering,
prediction, and grid sampling.

The optional experimental Lamperti--Euler sampler uses

$$
y=\frac{2}{\xi}\arcsin\sqrt{\tau},
\qquad
\tau=\sin^2\left(\frac{\xi y}{2}\right)
$$

and the unit-diffusion drift

$$
b_y(\tau)=
\frac{\kappa(m-\tau)}{\xi\sqrt{\tau(1-\tau)}}
-\frac{\xi(1-2\tau)}{4\sqrt{\tau(1-\tau)}}.
$$

With `S` substeps, $h=1/((n-1)S)$ and
$y_{j+1}=y_j+b_y(\tau_j)h+\sqrt{h}Z_j$. Drift evaluation uses an explicit
interior epsilon because the formula is singular at the endpoints. Overshoots
are handled by the configured reflection or clipping policy and counted in
sampling diagnostics. The initial value still follows the exact stationary
beta law. This path is an approximate sampling oracle only: likelihood,
gradient, filtering, and prediction continue to use their configured
transition backend.

The optimized implementation keeps this recursion in a strictly sequential
Numba kernel. `parallel=True` is forbidden because it would violate the causal
state update. The kernel never owns an RNG: Python draws stationary beta and
Gaussian values from the supplied `numpy.random.Generator`, passes Gaussian
values in complete-interval chunks, and carries the final transformed state
between chunks. Python and Numba executions must agree pathwise on identical
innovations, including intervention counts.

Stationary shapes below one are reported by
`stationary_boundary_singular=True`. This is a diagnostic, not an accuracy
guarantee: extreme asymmetric boundary-singular laws can retain material
reflection bias as the number of substeps grows.

## Static Elliptical Correlation Estimation

For static `GaussianCopula` and `StudentCopula`, `method="mle"` identifies the
static model/result contract. Correlation treatment is selected separately by
`corr_mode`; therefore an MLE-labelled result may contain a supplied or
plug-in correlation that was not optimized jointly with the other model
parameters.

In `fixed` mode, a supplied $R$ is evaluated unchanged. If $R$ is omitted,
Gaussian estimates $R$ from normal scores, while Student maps pairwise Kendall
statistics by

$$R_{ij}=\sin\left(\frac{\pi\tau_{ij}}{2}\right)$$

and projects to a valid SPD correlation when necessary. These plug-in
correlations are counted in AIC/BIC because they are estimated from the same
sample, even though they are absent from the optimizer vector.

`shrinkage` uses

$$R(\alpha)=\alpha R_0 + (1-\alpha)I,\qquad 0<\alpha<1,$$

and jointly optimizes one raw logit parameter. `cholesky` maps
$d(d-1)/2$ unconstrained raw values to a full SPD correlation and jointly
optimizes them. The latter is intended for small $d$. Factor mode represents

$$R=D+BB^\top,\qquad D_{ii}=1-\lVert B_{i\cdot}\rVert^2,$$

with identifiable count $dk-k(k-1)/2$. Two-stage factor fits count the
estimated loadings as plug-in parameters. Joint static Student factor fitting
optimizes `df` and identified loadings together; Gaussian factor fitting is
two-stage.

Consequently, with $q=d(d-1)/2$ and
$f=dk-k(k-1)/2$, the effective counts are:

| Correlation policy | Gaussian | Student |
|---|---:|---:|
| supplied `fixed` | $0$ | $1$ |
| plug-in `fixed` | $q$ | $1+q$ |
| `shrinkage` | $1$ | $2$ |
| `cholesky` | $q$ | $1+q$ |
| factor two-stage | $f$ | $1+f$ |
| factor joint | unavailable | $1+f$ |

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
