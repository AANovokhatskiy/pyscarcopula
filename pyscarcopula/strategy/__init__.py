"""
pyscarcopula.strategy — estimation methods for copula models.

Each method is a separate module with a class decorated by
@register_strategy('METHOD_NAME'). Use get_strategy() to obtain
an instance by method name.

Available methods:
    'MLE'        — constant parameter (strategy/mle.py)
    'SCAR-TM-OU' — transfer matrix with OU latent (strategy/scar_tm.py)
    'SCAR-P-OU'  — MC p-sampler with OU latent (strategy/scar_mc.py)
    'SCAR-M-OU'  — MC m-sampler with EIS (strategy/scar_mc.py)
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
)

__all__ = ['FitStrategy', 'get_strategy', 'register_strategy', 'list_methods']
