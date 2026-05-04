import numpy as np
import pytest

from pyscarcopula.contrib.marginal import (
    ARMAGARCHMarginal,
    MarginalModel,
    arma_garch_pobs,
)


def test_arma_garch_pobs_shape_and_bounds():
    rng = np.random.default_rng(123)
    data = rng.standard_normal((120, 2))

    u, fits = arma_garch_pobs(data, maxiter=80)

    assert u.shape == data.shape
    assert np.all(np.isfinite(u))
    assert np.all((u > 0.0) & (u < 1.0))
    assert len(fits) == 2
    assert all(fit.params.shape == (5,) for fit in fits)
    assert all(np.all(np.isfinite(fit.standardized_residuals)) for fit in fits)


def test_arma_garch_factory_alias():
    model = MarginalModel.create("arma-garch")

    assert isinstance(model, ARMAGARCHMarginal)


def test_arma_garch_static_distribution_methods_are_explicitly_unsupported():
    model = ARMAGARCHMarginal()

    with pytest.raises(NotImplementedError):
        model.ppf(np.array([[0.5]]), np.zeros((1, 5)))
