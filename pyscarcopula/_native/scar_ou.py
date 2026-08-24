"""SCAR-OU facade over the transitional numerical adapter."""

from importlib import import_module


def _adapter():
    return import_module("pyscarcopula.numerical._cpp_scar_ou")


def __getattr__(name):
    return getattr(_adapter(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_adapter())))
