"""Compare static elliptical fits with copulae on the crypto6 dataset.

This is an opt-in validation tool.  ``copulae`` is intentionally not a
runtime dependency of pyscarcopula::

    python -m pip install copulae pandas
    python tools/compare_copulae_crypto6.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from pyscarcopula import GaussianCopula, StudentCopula
from pyscarcopula._utils import pobs
from pyscarcopula.stattests import (
    cvm_test,
    gaussian_rosenblatt_transform,
    gof_test,
    student_rosenblatt_transform,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"


@dataclass(frozen=True)
class ComparisonRow:
    family: str
    implementation: str
    log_likelihood: float
    gof_statistic: float
    gof_pvalue: float
    df: float | None
    corr_max_abs_delta_from_copulae: float
    corr_frobenius_delta_from_copulae: float
    elapsed_seconds: float


def load_crypto6() -> tuple[pd.DataFrame, np.ndarray]:
    columns = [
        "BTC-USD", "ETH-USD", "BNB-USD",
        "ADA-USD", "XRP-USD", "DOGE-USD",
    ]
    prices = pd.read_csv(DATA_DIR / "crypto_prices.csv", index_col=0, sep=";")
    returns = np.log(prices[columns] / prices[columns].shift(1))
    returns = returns.iloc[1:].dropna().iloc[:250]
    return returns, pobs(returns.to_numpy(dtype=np.float64))


def _copulae_classes():
    try:
        from copulae import GaussianCopula as CopulaeGaussianCopula
        from copulae import StudentCopula as CopulaeStudentCopula
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "copulae is required: python -m pip install copulae") from exc
    return CopulaeGaussianCopula, CopulaeStudentCopula


def _fit_copulae(u: np.ndarray):
    CopulaeGaussianCopula, CopulaeStudentCopula = _copulae_classes()

    started = perf_counter()
    gaussian = CopulaeGaussianCopula(dim=u.shape[1])
    gaussian.fit(u, method="ml", to_pobs=False, verbose=0)
    gaussian_elapsed = perf_counter() - started

    started = perf_counter()
    student = CopulaeStudentCopula(dim=u.shape[1])
    student.fit(
        u,
        method="ml",
        to_pobs=False,
        fix_df=False,
        verbose=0,
    )
    student_elapsed = perf_counter() - started
    return gaussian, gaussian_elapsed, student, student_elapsed


def _correlation_delta(
        correlation: np.ndarray,
        oracle_correlation: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(correlation) - np.asarray(oracle_correlation)
    return float(np.max(np.abs(delta))), float(np.linalg.norm(delta, ord="fro"))


def _pyscar_row(
        family: str,
        mode: str,
        u: np.ndarray,
        oracle_correlation: np.ndarray,
        *,
        gtol: float,
        maxiter: int) -> ComparisonRow:
    model = (
        GaussianCopula(corr_mode=mode)
        if family == "gaussian"
        else StudentCopula(corr_mode=mode)
    )
    fit_kwargs = (
        {
            "gtol": gtol,
            "ftol": 1e-15,
            "maxiter": maxiter,
            "maxls": 100,
        }
        if mode == "cholesky"
        else {}
    )
    started = perf_counter()
    result = model.fit(u, to_pobs=False, **fit_kwargs)
    elapsed = perf_counter() - started
    if not result.success:
        raise RuntimeError(f"{family}/{mode} fit failed: {result.message}")
    gof = gof_test(model, u, fit_result=result, to_pobs=False)
    maximum, frobenius = _correlation_delta(
        result.correlation_matrix, oracle_correlation)
    return ComparisonRow(
        family=family,
        implementation=f"pyscarcopula-{mode}",
        log_likelihood=float(result.log_likelihood),
        gof_statistic=float(gof.statistic),
        gof_pvalue=float(gof.pvalue),
        df=None if family == "gaussian" else float(model.df),
        corr_max_abs_delta_from_copulae=maximum,
        corr_frobenius_delta_from_copulae=frobenius,
        elapsed_seconds=elapsed,
    )


def _copulae_row(
        family: str,
        model,
        elapsed: float,
        u: np.ndarray) -> ComparisonRow:
    correlation = np.asarray(model.sigma, dtype=np.float64)
    if family == "gaussian":
        transformed = gaussian_rosenblatt_transform(correlation, u)
        df = None
    else:
        df = float(model.params.df)
        transformed = student_rosenblatt_transform(correlation, df, u)
    gof = cvm_test(transformed)
    return ComparisonRow(
        family=family,
        implementation="copulae-joint-ml",
        log_likelihood=float(model.log_lik(u, to_pobs=False)),
        gof_statistic=float(gof.statistic),
        gof_pvalue=float(gof.pvalue),
        df=df,
        corr_max_abs_delta_from_copulae=0.0,
        corr_frobenius_delta_from_copulae=0.0,
        elapsed_seconds=elapsed,
    )


def compare(*, gtol: float, maxiter: int) -> pd.DataFrame:
    _, u = load_crypto6()
    copulae_gaussian, gaussian_elapsed, copulae_student, student_elapsed = (
        _fit_copulae(u))
    oracle_correlations = {
        "gaussian": np.asarray(copulae_gaussian.sigma, dtype=np.float64),
        "student": np.asarray(copulae_student.sigma, dtype=np.float64),
    }
    rows = [
        _pyscar_row(
            family, mode, u, oracle_correlations[family],
            gtol=gtol, maxiter=maxiter)
        for family in ("gaussian", "student")
        for mode in ("fixed", "cholesky")
    ]
    rows.extend((
        _copulae_row(
            "gaussian", copulae_gaussian, gaussian_elapsed, u),
        _copulae_row("student", copulae_student, student_elapsed, u),
    ))
    frame = pd.DataFrame(row.__dict__ for row in rows)
    return frame.sort_values(
        ["family", "implementation"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument(
        "--csv", type=Path,
        help="optional path for the full-precision result table")
    args = parser.parse_args()
    result = compare(gtol=args.gtol, maxiter=args.maxiter)
    with pd.option_context(
            "display.max_columns", None, "display.width", 180,
            "display.precision", 8):
        print(result.to_string(index=False))
    if args.csv is not None:
        result.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
