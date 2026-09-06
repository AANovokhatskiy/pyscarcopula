"""
pyscarcopula.strategy — estimation methods for copula models.

Each method is a separate module with a class decorated by
@register_strategy('METHOD_NAME'). Use get_strategy() to obtain
an instance by method name.

Available methods:
    'SCAR-TM-JACOBI' - transfer matrix with Jacobi Kendall-tau latent
    'MLE'        — constant parameter (strategy/mle.py)
    'SCAR-TM-OU' — transfer matrix with OU latent (strategy/scar_tm.py)
    'GAS'        — score-driven (strategy/gas.py)

Usage:
    from pyscarcopula.strategy import get_strategy, list_methods

    strategy = get_strategy('scar-tm-ou')
    result = strategy.fit(copula, u)
"""

from pyscarcopula.strategy._base import (
    FitStrategy,
    get_strategy,
    register_strategy,
    list_methods,
    validate_strategy_method,
)
from pyscarcopula.strategy.multivariate_mle import (
    StaticMLEEvaluation,
    StaticMLEOutcome,
    StaticMLEProblem,
    run_static_multivariate_mle,
)

__all__ = [
    'FitStrategy',
    'get_strategy',
    'register_strategy',
    'list_methods',
    'validate_strategy_method',
    'StaticMLEEvaluation',
    'StaticMLEOutcome',
    'StaticMLEProblem',
    'run_static_multivariate_mle',
]
