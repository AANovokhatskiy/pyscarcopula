"""Student quantile oracle independent of SciPy's changing stdtrit backend."""

import numpy as np
from scipy.special import betaincinv
from scipy.stats import beta


def student_quantile_beta_oracle(df, probabilities):
    """Invert the Student/beta identity with the native finite-tail policy.

    Older SciPy stdtrit uses CDFLIB's 1e100 search bound and looser root
    tolerances. Neither defines the native quantile contract. Use the inverse
    regularized beta function, flooring its tail argument at float64.tiny.
    The complementary identity avoids cancellation around the median.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    tail = np.minimum(probabilities, 1.0 - probabilities)
    magnitude = np.empty_like(tail)
    central = (tail >= 0.25) & (df >= 0.1)
    z = betaincinv(0.5, 0.5 * df, 1.0 - 2.0 * tail[central])
    magnitude[central] = np.sqrt(df * z / (1.0 - z))
    z = betaincinv(0.5 * df, 0.5, 2.0 * tail[~central])
    z = np.maximum(z, np.finfo(np.float64).tiny)
    # Compute 1-z directly: subtracting from one loses precision at large df.
    # beta.isf exposes this inverse complement even on supported SciPy 1.9.
    complement = beta.isf(2.0 * tail[~central], 0.5, 0.5 * df)
    magnitude[~central] = np.sqrt(df * complement / z)
    return np.copysign(magnitude, probabilities - 0.5)
