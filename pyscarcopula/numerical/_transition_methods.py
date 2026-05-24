"""Transition-method normalization shared by numerical backends."""

OU_TRANSITION_METHODS = frozenset(("auto", "matrix", "local", "spectral"))
JACOBI_MATRIX_TRANSITION_METHODS = frozenset(
    ("auto", "spectral_matrix", "local", "local_fixed")
)
JACOBI_STRATEGY_TRANSITION_METHODS = frozenset(
    ("auto", "spectral_matrix", "local", "local_fixed", "spectral_coeff")
)


def normalize_ou_transition_method(value):
    method = str(value).lower()
    if method not in OU_TRANSITION_METHODS:
        raise ValueError(
            "transition_method must be one of "
            "'auto', 'matrix', 'local', or 'spectral'")
    return method


def normalize_ou_grid_transition_method(value):
    method = normalize_ou_transition_method(value)
    if method == "spectral":
        # Spectral likelihood has no finite grid state; grid-only routines use
        # the automatic matrix/local fallback for forward distributions.
        return "auto"
    return method


def normalize_jacobi_matrix_transition_method(value):
    method = str(value).lower().replace("-", "_")
    aliases = {
        "matrix": "spectral_matrix",
        "spectral": "spectral_matrix",
    }
    method = aliases.get(method, method)
    if method not in JACOBI_MATRIX_TRANSITION_METHODS:
        raise ValueError(
            "transition_method must be one of "
            "'auto', 'spectral_matrix', 'local', or 'local_fixed'")
    return method


def normalize_jacobi_strategy_transition_method(value):
    method = str(value).lower().replace("-", "_")
    aliases = {
        "matrix": "spectral_matrix",
        "spectral": "spectral_coeff",
    }
    method = aliases.get(method, method)
    if method not in JACOBI_STRATEGY_TRANSITION_METHODS:
        raise ValueError(
            "transition_method must be one of 'auto', "
            "'spectral_matrix', 'local', 'local_fixed', "
            "or 'spectral_coeff'")
    return method
