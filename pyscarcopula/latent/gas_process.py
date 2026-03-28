"""
Backward compatibility — all GAS filter code now lives in
pyscarcopula.numerical.gas_filter.

This module re-exports the old names so that existing imports
continue to work.
"""

from pyscarcopula.numerical.gas_filter import (
    gas_filter as _gas_filter_full,
    gas_negloglik as _gas_loglik,
    gas_rosenblatt as _gas_rosenblatt,
    gas_mixture_h as _gas_mixture_h,
)

__all__ = ['_gas_filter_full', '_gas_loglik', '_gas_rosenblatt', '_gas_mixture_h']
