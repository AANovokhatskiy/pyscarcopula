import numpy as np


def test_risk_metrics_forwards_n_jobs_to_marginal_fit(monkeypatch):
    from pyscarcopula.contrib import risk_metrics as risk_module
    from pyscarcopula.contrib.marginal import MarginalModel

    calls = {}

    class RecordingMarginal:
        def fit_rolling(self, data, window_len, n_jobs=-1):
            calls["marginal_n_jobs"] = n_jobs
            return np.zeros((data.shape[0], data.shape[1], 1))

    class DummyCopula:
        _rotate = 0

    data = np.arange(12, dtype=np.float64).reshape(6, 2)
    expected_weight = np.array([0.5, 0.5])

    monkeypatch.setattr(
        MarginalModel,
        "create",
        staticmethod(lambda name: RecordingMarginal()),
    )

    def fake_calculate_fixed(
            copula, data, method, marginal_model, marg_params, gamma,
            window_len, N_mc, portfolio_weight, n_jobs=1,
            window_seed_sequences=None, **kwargs):
        calls["cvar_n_jobs"] = n_jobs
        np.testing.assert_allclose(portfolio_weight, expected_weight)
        return (
            np.zeros(data.shape[0]),
            np.zeros(data.shape[0]),
            portfolio_weight,
        )

    monkeypatch.setattr(
        risk_module,
        "_calculate_cvar_fixed",
        fake_calculate_fixed,
    )

    risk_module.risk_metrics(
        DummyCopula(),
        data,
        window_len=3,
        gamma=0.9,
        N_mc=10,
        marginals_method="johnsonsu",
        optimize_portfolio=False,
        n_jobs=4,
        rng=123,
    )

    assert calls == {"marginal_n_jobs": 4, "cvar_n_jobs": 4}
