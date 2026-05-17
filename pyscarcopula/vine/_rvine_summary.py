"""Text formatting helpers for ``RVineCopula`` summaries."""

import numpy as np

from pyscarcopula.vine._edge_adapter import (
    edge_copula,
    edge_has_dynamic_params,
    edge_is_independent,
    edge_param,
    edge_result,
    result_param_items,
)


def _family_name(copula):
    name = type(copula).__name__
    if name == 'BivariateGaussianCopula':
        return 'GaussianCopula'
    return name


def _named_float(value):
    value = float(value)
    if abs(value) < 5e-4:
        value = 0.0
    return f"{value:7.3f}"


def _dynamic_params(pc):
    if edge_is_independent(pc):
        return ''
    result = edge_result(pc)
    return ", ".join(
        f"{name}={_named_float(value)}"
        for name, value in result_param_items(result))


def _scalar_param(pc):
    param = edge_param(pc, default=0.0)
    if edge_is_independent(pc):
        return f"{param:.4f}"
    if edge_has_dynamic_params(pc):
        return ''
    return f"{param:.4f}"


def format_rvine_summary(vine):
    """Return a human-readable summary for a fitted or unfitted R-vine."""
    if vine.matrix is None:
        return "RVineCopula (unfitted)"

    lines = []
    lines.append(
        f"RVineCopula(d={vine.d}, T={vine._T}, "
        f"criterion={vine.criterion!r})"
    )
    lines.append(f"  log_likelihood = {vine._log_likelihood:.4f}")
    lines.append(f"  n_parameters   = {vine.n_parameters}")
    lines.append(f"  AIC = {vine.aic:.4f}   BIC = {vine.bic:.4f}")
    if vine.truncation_level is not None:
        lines.append(f"  truncation_level = {vine.truncation_level}")
        lines.append(f"  truncation_fill  = {vine.truncation_fill}")
    if vine.threshold not in (None, 0.0):
        lines.append(f"  threshold        = {vine.threshold}")
    if vine.min_edge_logL is not None:
        lines.append(f"  min_edge_logL   = {vine.min_edge_logL}")

    lines.append("")
    lines.append(
        "Structure matrix (natural order, Czado 2019 Alg. 5.4; "
        "anti-diagonal = leaf peeled at each column):"
    )
    lines.append(np.array2string(vine.matrix, separator=" "))

    lines.append("")
    lines.append("Edges (tree t, column col):")
    has_dynamic = any(
        edge_has_dynamic_params(pc)
        for pc in vine.pair_copulas.values())
    if has_dynamic:
        header = f"  {'t':>2} {'col':>4} {'pair':>10} {'cond':>14}  "\
                 f"{'family':<18} {'rot':>4}  {'dyn_params':<45}"\
                 f"{'param':>9} {'tau':>7} {'logL':>10}"
    else:
        header = f"  {'t':>2} {'col':>4} {'pair':>10} {'cond':>14}  "\
                 f"{'family':<18} {'rot':>4} {'param':>9} "\
                 f"{'tau':>7} {'logL':>10}"
    lines.append(header)
    d = vine.d
    for t in range(d - 1):
        for col in range(d - 1 - t):
            pc = vine.pair_copulas[(t, col)]
            leaf = int(vine.matrix[d - 1 - col, col])
            tail = int(vine.matrix[d - 2 - col - t, col])
            cond = sorted(
                int(vine.matrix[r, col])
                for r in range(d - 1 - col - t, d - 1 - col)
            )
            pair_str = f"({leaf},{tail})"
            cond_str = ",".join(str(c) for c in cond) if cond else "-"
            copula = edge_copula(pc)
            fam = _family_name(copula)
            rot = int(getattr(copula, 'rotate', 0))
            param = _scalar_param(pc)
            base = (
                f"  {t:>2} {col:>4} {pair_str:>10} {cond_str:>14}  "
                f"{fam:<18} {rot:>4} "
            )
            if has_dynamic:
                dyn_params = _dynamic_params(pc)
                lines.append(
                    f"{base} {dyn_params:<45} {param:>9} "
                    f"{pc.tau:>7.3f} {pc.log_likelihood:>10.3f}"
                )
            else:
                lines.append(
                    f"{base} {param:>9} "
                    f"{pc.tau:>7.3f} {pc.log_likelihood:>10.3f}"
                )
    return "\n".join(lines)
