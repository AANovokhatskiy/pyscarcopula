"""Golden contract for the production bivariate GAS implementation."""

import numpy as np
import pytest

from pyscarcopula.copula.clayton import ClaytonCopula
from pyscarcopula.copula.elliptical import BivariateGaussianCopula
from pyscarcopula.copula.frank import FrankCopula
from pyscarcopula.copula.gumbel import GumbelCopula
from pyscarcopula.copula.independent import IndependentCopula
from pyscarcopula.copula.joe import JoeCopula
from pyscarcopula.numerical import _cpp_gas
from pyscarcopula.numerical.gas_filter import (
    gas_filter,
    gas_mixture_h,
    gas_negloglik,
    gas_predict_param,
    gas_rosenblatt,
)


PARAMS = (0.07, 0.35, 0.62)
SCORE_EPS = 1e-4
GOLDEN_SCALINGS = [
    "unit",
    pytest.param("fisher", marks=pytest.mark.sanitizer_numerical),
]
OBSERVATIONS = np.array(
    [
        [0.12, 0.83],
        [0.71, 0.28],
        [0.44, 0.62],
        [0.91, 0.17],
        [0.33, 0.76],
        [0.58, 0.39],
    ],
    dtype=np.float64,
)


COPULA_FACTORIES = {
    "independent": IndependentCopula,
    "gaussian": BivariateGaussianCopula,
    "clayton_r0": lambda: ClaytonCopula(rotate=0),
    "clayton_r90": lambda: ClaytonCopula(rotate=90),
    "clayton_r180": lambda: ClaytonCopula(rotate=180),
    "clayton_r270": lambda: ClaytonCopula(rotate=270),
    "clayton_xtanh": lambda: ClaytonCopula(transform_type="xtanh"),
    "frank": FrankCopula,
    "frank_xtanh": lambda: FrankCopula(transform_type="xtanh"),
    "gumbel_r0": lambda: GumbelCopula(rotate=0),
    "gumbel_r90": lambda: GumbelCopula(rotate=90),
    "gumbel_r180": lambda: GumbelCopula(rotate=180),
    "gumbel_r270": lambda: GumbelCopula(rotate=270),
    "gumbel_xtanh": lambda: GumbelCopula(transform_type="xtanh"),
    "joe_r0": lambda: JoeCopula(rotate=0),
    "joe_r90": lambda: JoeCopula(rotate=90),
    "joe_r180": lambda: JoeCopula(rotate=180),
    "joe_r270": lambda: JoeCopula(rotate=270),
    "joe_xtanh": lambda: JoeCopula(transform_type="xtanh"),
}


# Tuple fields: logL, final g, final r, final clipped score, current r, next r.
GOLDEN_SUMMARIES = {
    "independent": {
        "unit": (0.0, 0.1842105263157895, 0.0, 0.0, 0.0, 0.0),
        "fisher": (0.0, 0.1842105263157895, 0.0, 0.0, 0.0, 0.0),
    },
    "gaussian": {
        "unit": (-0.10428019827856073, 0.06202469714462267,
                 0.015503381136877055, -0.010688520856052326,
                 0.015503381136877055, 0.026169986641500838),
        "fisher": (-493.78513569147515, 4.014951636476619,
                   0.7630832024085431, 2.9925029342164033,
                   0.7630832024085431, 0.7170342154246585),
    },
    "clayton_r0": {
        "unit": (-1.5976602481121041, -0.0032631146772172243,
                 0.6916169542104208, 0.05728473945282521,
                 0.6916169542104208, 0.7382287155551288),
        "fisher": (-55.47878722510957, -8.28141861244155,
                   0.0003531457399179634, 6.754022896515721,
                   0.0003531457399179634, 0.0651075805060905),
    },
    "clayton_r90": {
        "unit": (1.8587668674309, 0.33329964092669595,
                 0.8737192799684786, 0.15535348304813512,
                 0.8737192799684786, 0.8723915866490005),
        "fisher": (2.346524712945886, -7.600923328524393,
                   0.0005998646124832997, 0.15320771144278608,
                   0.0005998646124832997, 0.010212225534708689),
    },
    "clayton_r180": {
        "unit": (-1.8273044324505112, -0.06397898069042207,
                 0.6617692667182177, 0.053567842170961194,
                 0.6617692667182177, 0.7180891663056356),
        "fisher": (-74.11402920938332, -8.37440504683663,
                   0.00033067047424907694, 15.465084768351289,
                   0.00033067047424907694, 0.849094062024069),
    },
    "clayton_r270": {
        "unit": (1.8284163500244341, 0.34526144862520064,
                 0.8807051611287363, 0.10484601105722217,
                 0.8807051611287363, 0.8664322520442677),
        "fisher": (2.9583251919434703, 1.0367956480653122,
                   1.3403938030529552, 100.0, 1.3403938030529552,
                   35.7129133018005),
    },
    "clayton_xtanh": {
        "unit": (-0.04491353319654223, 0.11967899647608936,
                 0.014355068352760285, 0.007268647040423035,
                 0.014355068352760285, 0.02148084377319515),
        "fisher": (-70.14500341499483, 26.918380446929106,
                   26.918480446929106, -100.0, 26.918480446929106,
                   18.240704122903946),
    },
    "frank": {
        "unit": (-0.5869376373705972, 0.11805637724347229,
                 0.7540165219354549, 0.003112317942607756,
                 0.7540165219354549, 0.7679893026961429),
        "fisher": (-0.26462349003472907, -13.494812506054846,
                   0.00010137808845703611, -0.02424727085781342,
                   0.00010137808845703611, 0.00034717996310125606),
    },
    "frank_xtanh": {
        "unit": (-0.020390392073672103, 0.14891539180813523,
                 0.022113313076691787, -0.00495410914771254,
                 0.022113313076691787, 0.025670856176594422),
        "fisher": (-0.013335581563130638, 0.09536367658276407,
                   0.009166762384060249, -0.09996918756057022,
                   0.009166762384060249, 0.00893555242426631),
    },
    "gumbel_r0": {
        "unit": (-3.324762716775992, -0.23690046094139738,
                 1.5817958352777102, 0.09971092604636078,
                 1.5817958352777102, 1.672477717953289),
        "fisher": (-100.20469448771938, -8.470387613336877,
                   1.0003095616906137, 39.656915529207026,
                   1.0003095616906137, 9.698546973597873),
    },
    "gumbel_r90": {
        "unit": (3.295099272444696, 0.39824559930568415,
                 1.9120652841479027, 0.23410958694904938,
                 1.9120652841479027, 1.9124272956802169),
        "fisher": (5.876252261486459, 1.7907816916287271,
                   2.9451721125174175, 10.416884749440134,
                   2.9451721125174175, 5.834279319301897),
    },
    "gumbel_r180": {
        "unit": (-3.1529812204331753, -0.19498564267806251,
                 1.6004992748395237, 0.10234644423742553,
                 1.6004992748395237, 1.6857406463236024),
        "fisher": (-87.5559151346306, -8.383733897922324,
                   1.0003285288346608, 37.07914086881203,
                   1.0003285288346608, 8.850274047457782),
    },
    "gumbel_r270": {
        "unit": (3.24681493048096, 0.3724300242414399,
                 1.8967009227104241, 0.2743401357365246,
                 1.8967009227104241, 1.9112758201457056),
        "fisher": (6.619891320993255, 35.54250915133977,
                   36.542609151339775, -50.32084352194237,
                   36.542609151339775, 5.505273635634797),
    },
    "gumbel_xtanh": {
        "unit": (-0.0808095875096409, 0.05970788183982341,
                 1.0036608007039942, 0.022690405950527938,
                 1.0036608007039942, 1.0132580091147037),
        "fisher": (-126.30834622983643, 26.91874060836648,
                   27.91884060836648, -100.0, 27.91884060836648,
                   19.240480822812774),
    },
    "joe_r0": {
        "unit": (-1.9527469555251575, -0.09341570992747315,
                 1.647629741060952, 0.05237266447668028,
                 1.647629741060952, 1.7085691387928654),
        "fisher": (-74.18834778862018, -8.343926430147596,
                   1.0003378083795096, 44.432923451687145,
                   1.0003378083795096, 11.44841781882993),
    },
    "joe_r90": {
        "unit": (2.1604245569439233, 0.35533992112804053,
                 1.8866181033299116, 0.11198356635517069,
                 1.8866181033299116, 1.8715104171189418),
        "fisher": (3.7137782931326386, 2.4233258330608693,
                   3.50834252378675, 100.0, 3.50834252378675,
                   37.57256201649774),
    },
    "joe_r180": {
        "unit": (-1.6792292546728231, -0.021408775735093817,
                 1.6826000835581185, 0.05405863186246945,
                 1.6826000835581185, 1.7317858602596354),
        "fisher": (-55.57302118884707, -8.249538867533614,
                   1.0003613449063309, 40.24660604784458,
                   1.0003613449063309, 10.041816393377557),
    },
    "joe_r270": {
        "unit": (2.1575135625110264, 0.3308896964633482,
                 1.8723160441486304, 0.17148841726624442,
                 1.8723160441486304, 1.874810796699561),
        "fisher": (2.7834847187326353, 1.1961519598998978,
                   2.4604264711385726, 100.0, 2.4604264711385726,
                   36.81171421513794),
    },
    "joe_xtanh": {
        "unit": (-0.05215324683820065, 0.08459574732954467,
                 1.0072394176469854, 0.03111666872166006,
                 1.0072394176469854, 1.0177749807832408),
        "fisher": (-91.1160431572702, 26.919858797491756,
                   27.919958797491756, -100.0, 27.919958797491756,
                   19.239787545555103),
    },
}


GAUSSIAN_PATHS = {
    "unit": {
        "g": [
            0.1842105263157895,
            0.08048791349375078,
            0.09228333400542779,
            0.12495994208863705,
            0.030655234477493506,
            0.06202469714462267,
        ],
        "r": [
            0.04601550033499083,
            0.02011725113070182,
            0.02306443444542554,
            0.03122670375578703,
            0.00766289221509073,
            0.015503381136877055,
        ],
        "score": [
            -0.296350322348682,
            -0.07891192103056485,
            -0.006444928556366241,
            -0.3337712274784614,
            -0.07709013780406661,
            -0.010688520856052326,
        ],
    },
    "fisher": {
        "g": [
            0.1842105263157895,
            -0.9153205049269116,
            -35.49749871305468,
            13.061550797906097,
            7.419320790675862,
            4.014951636476619,
        ],
        "r": [
            0.04601550033499083,
            -0.22489546107044292,
            -0.9998999608436097,
            0.9969887858079696,
            0.9521035583359647,
            0.7630832024085431,
        ],
        "score": [
            -3.1415172321220037,
            -100.0,
            100.0,
            -2.139544868645486,
            -1.8715064392640424,
            2.9925029342164033,
        ],
    },
}


def _score_path(copula, scaling, g_path, r_path):
    scores = []
    for t in range(len(OBSERVATIONS)):
        update = _cpp_gas.update_one(
            *PARAMS,
            g_path[t],
            OBSERVATIONS[t],
            copula,
            scaling,
            SCORE_EPS,
        )
        assert update.r == pytest.approx(r_path[t])
        scores.append(update.score)
    return np.asarray(scores)


@pytest.mark.parametrize("name", COPULA_FACTORIES)
@pytest.mark.parametrize("scaling", GOLDEN_SCALINGS)
def test_bivariate_gas_families_match_regression_summaries(name, scaling):
    copula = COPULA_FACTORIES[name]()
    g_path, r_path, log_likelihood = gas_filter(
        *PARAMS,
        OBSERVATIONS,
        copula,
        scaling=scaling,
        score_eps=SCORE_EPS,
    )
    final_score = _score_path(copula, scaling, g_path, r_path)[-1]
    current = gas_predict_param(
        *PARAMS, OBSERVATIONS, copula, scaling, SCORE_EPS, "current")
    next_param = gas_predict_param(
        *PARAMS, OBSERVATIONS, copula, scaling, SCORE_EPS, "next")

    actual = (
        log_likelihood,
        g_path[-1],
        r_path[-1],
        final_score,
        current,
        next_param,
    )
    np.testing.assert_allclose(
        actual,
        GOLDEN_SUMMARIES[name][scaling],
        rtol=2e-8,
        atol=2e-9,
    )


@pytest.mark.parametrize("scaling", GOLDEN_SCALINGS)
def test_gaussian_gas_full_paths_match_regression_values(scaling):
    copula = BivariateGaussianCopula()
    g_path, r_path, _ = gas_filter(
        *PARAMS,
        OBSERVATIONS,
        copula,
        scaling=scaling,
        score_eps=SCORE_EPS,
    )
    score_path = _score_path(copula, scaling, g_path, r_path)

    expected = GAUSSIAN_PATHS[scaling]
    np.testing.assert_allclose(
        g_path, expected["g"], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        r_path, expected["r"], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        score_path, expected["score"], rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("scaling", ["unit", "fisher"])
def test_final_observation_affects_loglik_and_next_but_not_filtered_state(
        scaling):
    copula = BivariateGaussianCopula()
    changed = OBSERVATIONS.copy()
    changed[-1] = [0.1, 0.9]

    g_a, r_a, ll_a = gas_filter(
        *PARAMS, OBSERVATIONS, copula, scaling, SCORE_EPS)
    g_b, r_b, ll_b = gas_filter(
        *PARAMS, changed, copula, scaling, SCORE_EPS)

    np.testing.assert_allclose(g_a, g_b, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(r_a, r_b, rtol=0.0, atol=0.0)
    assert ll_a != pytest.approx(ll_b)
    assert gas_predict_param(
        *PARAMS, OBSERVATIONS, copula, scaling, SCORE_EPS, "current"
    ) == pytest.approx(r_a[-1])
    assert gas_predict_param(
        *PARAMS, changed, copula, scaling, SCORE_EPS, "current"
    ) == pytest.approx(r_b[-1])
    assert gas_predict_param(
        *PARAMS, OBSERVATIONS, copula, scaling, SCORE_EPS, "next"
    ) != pytest.approx(
        gas_predict_param(
            *PARAMS, changed, copula, scaling, SCORE_EPS, "next")
    )


@pytest.mark.parametrize("scaling", ["unit", "fisher"])
def test_gamma_zero_keeps_stationary_state_constant(scaling):
    omega, gamma, beta = 0.08, 0.0, 0.6
    copula = GumbelCopula(rotate=90)

    g_path, r_path, _ = gas_filter(
        omega,
        gamma,
        beta,
        OBSERVATIONS,
        copula,
        scaling,
        SCORE_EPS,
    )

    expected_g = omega / (1.0 - beta)
    np.testing.assert_allclose(g_path, expected_g, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(r_path, r_path[0], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("scaling", GOLDEN_SCALINGS)
def test_gaussian_loglik_h_and_rosenblatt_match_regression_values(scaling):
    copula = BivariateGaussianCopula()
    expected_h = {
        "unit": [
            0.8435850859671127,
            0.2762245165407202,
            0.6213563594303475,
            0.15949912007564848,
            0.7610531693531034,
            0.3887865838482283,
        ],
        "fisher": [
            0.8435850860536873,
            0.31902067372327536,
            0.999999,
            1e-06,
            0.9998832193931417,
            0.25125169082774224,
        ],
    }[scaling]
    expected_log_likelihood = GOLDEN_SUMMARIES["gaussian"][scaling][0]

    h_path = gas_mixture_h(
        *PARAMS, OBSERVATIONS, copula, scaling, SCORE_EPS)
    rosenblatt = gas_rosenblatt(
        *PARAMS, OBSERVATIONS, copula, scaling, SCORE_EPS)

    np.testing.assert_allclose(
        h_path, expected_h, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        rosenblatt[:, 0],
        np.clip(OBSERVATIONS[:, 0], 1e-6, 1.0 - 1e-6),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rosenblatt[:, 1], expected_h, rtol=2e-8, atol=2e-9)
    assert gas_negloglik(
        *PARAMS, OBSERVATIONS, copula, scaling, SCORE_EPS
    ) == pytest.approx(-expected_log_likelihood, rel=2e-8, abs=2e-9)


@pytest.mark.parametrize(
    "copula",
    [
        BivariateGaussianCopula(),
        ClaytonCopula(),
        FrankCopula(),
        GumbelCopula(),
        JoeCopula(),
    ],
    ids=["gaussian", "clayton", "frank", "gumbel", "joe"],
)
@pytest.mark.parametrize("scaling", ["unit", "fisher"])
def test_near_boundary_pseudo_observations_remain_finite(copula, scaling):
    u = np.array(
        [
            [1e-12, 1.0 - 1e-12],
            [1.0 - 1e-12, 1e-12],
            [1e-10, 1e-10],
            [1.0 - 1e-10, 1.0 - 1e-10],
        ],
        dtype=np.float64,
    )

    g_path, r_path, log_likelihood = gas_filter(
        0.02, 0.2, 0.4, u, copula, scaling, SCORE_EPS)

    assert np.all(np.isfinite(g_path))
    assert np.all(np.isfinite(r_path))
    assert np.isfinite(log_likelihood)
