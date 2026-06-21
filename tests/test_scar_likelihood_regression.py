"""Regression values for deterministic SCAR-TM-OU calculations."""

import numpy as np
import pytest

from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.multivariate.stochastic_student import (
    StochasticStudentCopula,
)
from pyscarcopula.numerical import _cpp_scar_ou
from pyscarcopula.numerical._scar_ou_config import AutoTMConfig


PARAMS = (1.15, 0.25, 0.8)
CONFIG = AutoTMConfig(
    transition_method="matrix",
    K=16,
    max_K=16,
    adaptive=False,
    grid_range=4.0,
)
GRID = np.array(
    [
        -1.8600175148665188,
        -1.5786818462176497,
        -1.2973461775687805,
        -1.0160105089199112,
        -0.7346748402710421,
        -0.4533391716221731,
        -0.1720035029733038,
        0.10933216567556547,
        0.3906678343244345,
        0.6720035029733036,
        0.9533391716221726,
        1.2346748402710421,
        1.5160105089199112,
        1.7973461775687802,
        2.0786818462176497,
        2.360017514866519,
    ]
)


def _assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)


@pytest.mark.skipif(
    not _cpp_scar_ou.available(), reason="requires bundled C++ extension"
)
def test_bivariate_scar_matrix_matches_regression_values():
    observations = np.array(
        [
            [0.12, 0.83],
            [0.71, 0.28],
            [0.44, 0.62],
            [0.91, 0.17],
            [0.33, 0.76],
            [0.58, 0.39],
        ]
    )
    copula = ClaytonCopula(rotate=180, transform_type="softplus")

    log_likelihood, _ = _cpp_scar_ou.loglik(
        *PARAMS, observations, copula, CONFIG
    )
    negative, gradient = _cpp_scar_ou.neg_loglik_with_grad(
        *PARAMS, observations, copula, CONFIG
    )
    predictive = _cpp_scar_ou.predictive_mean(
        *PARAMS, observations, copula, CONFIG
    )
    mixture_h = _cpp_scar_ou.mixture_h(
        *PARAMS, observations, copula, CONFIG
    )
    current_grid, current_prob = _cpp_scar_ou.state_distribution(
        *PARAMS, observations, copula, CONFIG, horizon="current"
    )
    next_grid, next_prob = _cpp_scar_ou.state_distribution(
        *PARAMS, observations, copula, CONFIG, horizon="next"
    )

    assert log_likelihood == pytest.approx(
        -1.9357642859184567, rel=2e-7, abs=2e-8
    )
    assert negative == pytest.approx(
        1.9357642859184567, rel=2e-7, abs=2e-8
    )
    _assert_close(gradient, [0.12086848, 1.38960469, -0.22124341])
    _assert_close(
        predictive,
        [
            0.8591948437945168,
            0.793530690073546,
            0.7916270883968823,
            0.8132700571718529,
            0.7360782477481742,
            0.7447603123916569,
        ],
    )
    _assert_close(
        mixture_h,
        [
            0.9508275037313086,
            0.2125074253792386,
            0.7081481876596774,
            0.054384698267825084,
            0.8683825069201148,
            0.3863608235049928,
        ],
    )
    _assert_close(current_grid, GRID)
    _assert_close(next_grid, GRID)
    _assert_close(
        current_prob,
        [
            8.695514794326424e-05,
            0.0012918986859461494,
            0.0066204211599841505,
            0.02447547465351284,
            0.06656135340765759,
            0.13384832886636078,
            0.19909112516857944,
            0.21882566803717343,
            0.1774568279939782,
            0.10598912492125707,
            0.046534848391556395,
            0.014990894475731525,
            0.0035370671207918766,
            0.0006100413338613947,
            7.653508002352451e-05,
            3.4355556421888286e-06,
        ],
    )
    _assert_close(
        next_prob,
        [
            7.384559056065986e-05,
            0.0011063667725613004,
            0.0057396028705357545,
            0.02157401488164368,
            0.05993108403444143,
            0.12374335088527143,
            0.19009952512554118,
            0.21723797903932968,
            0.18455604749169902,
            0.11647522080496021,
            0.054562640724747535,
            0.018955851102147644,
            0.004879347272795663,
            0.0009288032765688897,
            0.00012981401902212165,
            6.506108173797507e-06,
        ],
    )


@pytest.mark.skipif(
    not _cpp_scar_ou.available(), reason="requires bundled C++ extension"
)
def test_multivariate_student_scar_matrix_matches_regression_values():
    observations = np.array(
        [
            [0.12, 0.83, 0.41],
            [0.71, 0.28, 0.64],
            [0.44, 0.62, 0.19],
            [0.91, 0.17, 0.52],
            [0.33, 0.76, 0.87],
            [0.58, 0.39, 0.24],
        ]
    )
    correlation = np.array(
        [
            [1.0, 0.35, -0.15],
            [0.35, 1.0, 0.25],
            [-0.15, 0.25, 1.0],
        ]
    )
    copula = StochasticStudentCopula(d=3, R=correlation)

    log_likelihood, _ = _cpp_scar_ou.loglik(
        *PARAMS, observations, copula, CONFIG
    )
    negative, gradient = _cpp_scar_ou.neg_loglik_with_grad(
        *PARAMS, observations, copula, CONFIG
    )
    predictive = _cpp_scar_ou.predictive_mean(
        *PARAMS, observations, copula, CONFIG
    )
    current_grid, current_prob = _cpp_scar_ou.state_distribution(
        *PARAMS, observations, copula, CONFIG, horizon="current"
    )
    next_grid, next_prob = _cpp_scar_ou.state_distribution(
        *PARAMS, observations, copula, CONFIG, horizon="next"
    )

    assert log_likelihood == pytest.approx(
        -1.7288935699547465, rel=2e-7, abs=2e-8
    )
    assert negative == pytest.approx(
        1.7288935699547467, rel=2e-7, abs=2e-8
    )
    _assert_close(
        gradient,
        [-4.64709660e-05, 2.42546895e-03, -2.43472988e-05],
    )
    _assert_close(
        predictive,
        [
            2.8590958437945178,
            2.860228337957916,
            2.8608979734963844,
            2.862270151378401,
            2.864097294017034,
            2.861583131857681,
        ],
    )
    _assert_close(current_grid, GRID)
    _assert_close(next_grid, GRID)
    _assert_close(
        current_prob,
        [
            3.216789575494858e-05,
            0.0005172351914213175,
            0.0029331647402535783,
            0.01222688749599664,
            0.03808289988672748,
            0.08902098107373572,
            0.15633863818912833,
            0.206363783673179,
            0.20480119207049174,
            0.1528595353122507,
            0.08582510942975481,
            0.036249482679809965,
            0.011507827845421304,
            0.0027337415756935853,
            0.0004778777168654953,
            2.9475223515188164e-05,
        ],
    )
    _assert_close(
        next_prob,
        [
            3.205420276783613e-05,
            0.0005153643826137496,
            0.002921609531152802,
            0.012175605799522558,
            0.037926228388304895,
            0.08870290342426307,
            0.15593895850734255,
            0.20613060797564242,
            0.20492012295625142,
            0.15322877089927414,
            0.0861866028165583,
            0.036461027712641494,
            0.01159089853061672,
            0.002756820365521889,
            0.00048259400509091015,
            2.983050243519974e-05,
        ],
    )
