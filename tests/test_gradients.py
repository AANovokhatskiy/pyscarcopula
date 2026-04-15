"""Test analytical gradients against finite differences."""
import itertools

import numpy as np
import pytest
from pyscarcopula import (
    GumbelCopula, ClaytonCopula, FrankCopula, JoeCopula,
)
from pyscarcopula._utils import broadcast
from pyscarcopula.numerical.tm_gradient import tm_loglik_with_grad
from pyscarcopula.numerical.tm_functions import tm_loglik


# ═══════════════════════════════════════════════════════════
# d(log c)/dr verification
# ═══════════════════════════════════════════════════════════

COPULA_R_RANGES = [
    (GumbelCopula, 0, [1.01, 1.5, 3.0, 10.0, 30.0]),
    (GumbelCopula, 180, [1.01, 1.5, 3.0, 10.0, 30.0]),
    (ClaytonCopula, 0, [0.01, 0.5, 2.0, 10.0, 50.0]),
    (FrankCopula, 0, [0.1, 1.0, 5.0, 20.0, 50.0]),
    (JoeCopula, 0, [1.01, 1.5, 3.0, 10.0, 30.0]),
    (JoeCopula, 180, [1.01, 1.5, 3.0, 10.0, 30.0]),
]

COPULA_R_IDS = [f"{c.__name__}-{r}" for c, r, _ in COPULA_R_RANGES]


@pytest.mark.parametrize("cls,rot,r_vals", COPULA_R_RANGES, ids=COPULA_R_IDS)
def test_dlogc_dr(cls, rot, r_vals):
    """Analytical d(log c)/dr matches finite differences."""
    cop = cls(rotate=rot)
    eps = 1e-6
    u_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_err = 0.0

    for r_val in r_vals:
        for u1_val in u_vals:
            for u2_val in u_vals:
                u1a, u2a, ra = broadcast(
                    np.array([u1_val]), np.array([u2_val]), np.array([r_val]))
                v1, v2 = cop._apply_rotation(u1a, u2a)
                ana = cop.dlog_pdf_dr_unrotated(v1, v2, ra)[0]
                lp = cop.log_pdf_unrotated(v1, v2, ra + eps)[0]
                lm = cop.log_pdf_unrotated(v1, v2, ra - eps)[0]
                num = (lp - lm) / (2 * eps)

                if np.isfinite(ana) and np.isfinite(num):
                    denom = max(abs(num), abs(ana), 1e-10)
                    rel_err = abs(ana - num) / denom
                    max_err = max(max_err, rel_err)

    assert max_err < 1e-3, f"Max rel error {max_err:.2e}"


# ═══════════════════════════════════════════════════════════
# pdf_and_grad_on_grid verification
# ═══════════════════════════════════════════════════════════

@pytest.mark.parametrize("cls,rot,r_vals", COPULA_R_RANGES, ids=COPULA_R_IDS)
def test_pdf_and_grad_on_grid(cls, rot, r_vals):
    """pdf_and_grad_on_grid_batch matches finite differences on x."""
    cop = cls(rotate=rot)
    x_grid = np.linspace(-3.0, 3.0, 30)
    eps = 1e-6
    u_test = np.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])

    fi, dfi = cop.pdf_and_grad_on_grid_batch(u_test, x_grid)
    fi_p = cop.copula_grid_batch(u_test, x_grid + eps)
    fi_m = cop.copula_grid_batch(u_test, x_grid - eps)
    dfi_num = (fi_p - fi_m) / (2.0 * eps)

    mask = np.abs(dfi_num) > 1e-10
    if mask.any():
        rel_err = np.abs(dfi[mask] - dfi_num[mask]) / np.abs(dfi_num[mask])
        assert rel_err.max() < 1e-3, f"Max rel error {rel_err.max():.2e}"


# ═══════════════════════════════════════════════════════════
# Full TM gradient verification
# ═══════════════════════════════════════════════════════════

THETA_VALS = [0.01, 3.0, 10.0]
MU_VALS = [-5.0, 0.01, 5.0]
NU_VALS = [0.01, 1.5, 5.0]
ALPHA_GRID = [np.array([t, m, n])
              for t, m, n in itertools.product(THETA_VALS, MU_VALS, NU_VALS)]
ALPHA_IDS = [f"th={a[0]}_mu={a[1]}_nu={a[2]}" for a in ALPHA_GRID]


@pytest.mark.parametrize("cls,rot", [
    (GumbelCopula, 180),
    (ClaytonCopula, 0),
    (FrankCopula, 0),
    (JoeCopula, 180),
], ids=["Gumbel-180", "Clayton-0", "Frank-0", "Joe-180"])
@pytest.mark.parametrize("alpha", ALPHA_GRID, ids=ALPHA_IDS)
def test_tm_loglik_gradient(cls, rot, alpha, crypto_data):
    """Full TM logL gradient matches numerical finite differences."""
    cop = cls(rotate=rot)
    u = crypto_data[:300]

    val, grad = tm_loglik_with_grad(*alpha, u, cop, K=80)

    if val >= 1e10 or not np.all(np.isfinite(grad)):
        pytest.skip("loglik returned FAIL or NaN for this parameter set")

    eps = 1e-5
    grad_num = np.zeros(3)
    for k in range(3):
        a_p = alpha.copy(); a_p[k] += eps
        a_m = alpha.copy(); a_m[k] -= eps
        vp = tm_loglik(*a_p, u, cls(rotate=rot), K=80)
        vm = tm_loglik(*a_m, u, cls(rotate=rot), K=80)
        grad_num[k] = (vp - vm) / (2 * eps)

    if not np.all(np.isfinite(grad_num)):
        pytest.skip("numerical gradient contains non-finite values")

    # Mixed tolerance: relative when |grad| >> 1, absolute when |grad| << 1
    abs_errs = np.abs(grad - grad_num)
    scale = np.maximum(np.abs(grad_num), 1.0)
    rel_errs = abs_errs / scale
    assert rel_errs.max() < 0.01, \
        f"Gradient mismatch: rel_errs={rel_errs}, grad={grad}, num={grad_num}"
