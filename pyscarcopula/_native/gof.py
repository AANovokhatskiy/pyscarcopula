"""GoF facade over the transitional Student/Rosenblatt adapter."""

from importlib import import_module


def _adapter():
    return import_module("pyscarcopula.numerical.student_gof")


def __getattr__(name):
    return getattr(_adapter(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_adapter())))
