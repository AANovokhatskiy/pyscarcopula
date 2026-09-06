"""Independent local Gaussian emission oracle for Student OU single-start policy."""
import numpy as np
import pytest
from pyscarcopula._native import scar_ou


def test_variance_score_matches_dense_gaussian_oracle():
    rng = np.random.default_rng(52)
    values = rng.normal(size=41)
    left, mu, right, step = scar_ou.student_initial_stencil(0.3)
    logs = np.column_stack([-0.5 * (values - x)**2 for x in (left, mu, right)])
    alpha, info = scar_ou.student_score_initial_point(logs, 5.0, mu, -12.0, step)
    rho = np.exp(-alpha[0] / (len(values) - 1))
    C = rho ** np.abs(np.arange(len(values))[:, None] - np.arange(len(values)))
    scores = values - mu
    Q = 0.5 * (scores @ C @ scores - len(values))
    I = np.mean(scores**2)
    F = 0.5 * I**2 * np.trace(C @ C)
    np.testing.assert_allclose(info['variance_score'], Q, atol=2e-8)
    np.testing.assert_allclose(info['variance_information'], F, rtol=2e-11)
    np.testing.assert_allclose(info['sigma_x'], np.clip(np.sqrt(max(F**-0.5, Q/F)), .01, 2))


def test_no_information_has_finite_bounded_interior_start():
    alpha, info = scar_ou.student_score_initial_point(np.zeros((30, 3)), 1000, 998, 0, .01)
    assert np.isfinite(alpha).all()
    assert info['sigma_x'] == 2
    assert info['variance_information'] == 0


@pytest.mark.parametrize('logs,step', [(np.zeros((1,3)), .001), (np.full((10,3),np.nan), .001), (np.zeros((10,3)),0), (np.zeros((10,3)),np.inf)])
def test_invalid_native_summary_inputs(logs, step):
    with pytest.raises((ValueError, RuntimeError)):
        scar_ou.student_score_initial_point(logs, 5, 3, 0, step)


def test_information_floor_shrinks_with_data_replication():
    scales=[]
    for n in (100,1000):
        scores=np.resize([1.,-1.],n)
        step=.001
        logs=np.column_stack([scores*(-step)-.5*step**2,np.zeros(n),scores*step-.5*step**2])
        _,info=scar_ou.student_score_initial_point(logs,5,3,0,step)
        assert info['variance_score']<0
        scales.append(info['sigma_x'])
    assert scales[1] < scales[0]


@pytest.mark.parametrize('mu', [np.finfo(float).max, -np.finfo(float).max, np.nan, np.inf])
def test_invalid_stencil_is_rejected(mu):
    with pytest.raises((ValueError, RuntimeError)):
        scar_ou.student_initial_stencil(mu)


def test_explicit_legacy_diffusion_reports_actual_stationary_scale():
    alpha, info = scar_ou.stochastic_student_initial_point(100, 5, 3, -10, nu=.1)
    np.testing.assert_allclose(info['sigma_x'], alpha[2] / np.sqrt(2 * alpha[0]))


def test_resolved_fit_initialization_retains_student_diagnostics():
    from types import SimpleNamespace
    from pyscarcopula import StochasticStudentCopula
    from pyscarcopula.strategy.initial_point import resolve_ou_initial_point
    copula = StochasticStudentCopula(3, R=np.eye(3))
    u = np.random.default_rng(10).uniform(.01, .99, size=(40, 3))
    static = SimpleNamespace(copula_param=5.0, log_likelihood=0.0)
    alpha, info = resolve_ou_initial_point(
        copula, u, config=None, smart_init=True, verbose=False,
        alpha0=None, initial_mle_result=static)
    assert info['scale_method'] == 'variance_score'
    assert info['df_mle'] == 5.0
    assert info['variance_information'] > 0.0
    assert np.isfinite(info['variance_score'])
    assert info['stationary_scale_floor'] > 0.0
    np.testing.assert_allclose(info['sigma_x'], alpha[2] / np.sqrt(2 * alpha[0]))
    assert info['mle_source'] == 'selection_result'


def test_overflowing_scale_request_is_rejected():
    with pytest.raises((ValueError, RuntimeError)):
        scar_ou.student_score_initial_point(
            np.zeros((100, 3)), 5, 3, 0, .001,
            maximum_stationary_scale=1e308)
