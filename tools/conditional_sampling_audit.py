"""Generate the conditional-sampling coverage inventory.

The inventory combines pytest collection node IDs with a source-level AST
scan.  It is intentionally an audit heuristic, not a coverage substitute.
By default, ``tests/conditional`` is excluded so the report can compare the
general suite with the dedicated conditional-sampling contracts.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
REGISTRY_PATH = TESTS / "conditional" / "support_matrix.json"
DEFAULT_OUTPUT = (
    ROOT / "benchmark_artifacts" /
    "conditional_sampling_inventory.json"
)

_CONDITIONAL_NAME_PATTERN = re.compile(
    r"conditional|given",
    re.IGNORECASE,
)

_CONDITIONAL_SOURCE_PATTERN = re.compile(
    r"sample_conditional\s*\(|\bgiven\s*=|conditional_method|"
    r"conditional_sampling|dag_mcmc|suffix_sampling|dynamic_conditioning",
    re.IGNORECASE,
)

_MODEL_PATTERNS = (
    ("bivariate-independent", (
        r"\bindependentcopula\b", r"\bindependent-r\d+",
    )),
    ("bivariate-gaussian", (
        r"\bbivariategaussiancopula\b", r"\bgaussian-r\d+",
    )),
    ("bivariate-clayton", (r"\bclaytoncopula\b", r"\bclayton\b")),
    ("bivariate-gumbel", (r"\bgumbelcopula\b", r"\bgumbel\b")),
    ("bivariate-frank", (r"\bfrankcopula\b", r"\bfrank\b")),
    ("bivariate-joe", (r"\bjoecopula\b", r"\bjoe\b")),
    ("multivariate-equicorr-gaussian", (r"\bequicorrgaussiancopula\b",)),
    ("multivariate-stochastic-student", (r"\bstochasticstudentcopula\b",)),
    ("multivariate-gaussian", (r"\bgaussiancopula\b",)),
    ("multivariate-student", (r"\bstudentcopula\b",)),
    ("vine-legacy-cvine", (r"\bcvinecopula\b", r"legacy[_ ]cvine")),
    ("vine-generic", (r"\brvinecopula\b", r"\bvinecopula\b", r"dag_mcmc")),
)

_METHOD_PATTERNS = (
    ("SCAR-TM-JACOBI", ("scar-tm-jacobi", "scar_tm_jacobi", "jacobi")),
    ("SCAR-TM-OU", ("scar-tm-ou", "scar_tm_ou", "scar tm")),
    ("GAS", ("gas",)),
    ("MLE", ("mle",)),
)

_FEATURE_PATTERNS = (
    ("analytical_oracle", ("oracle", "closed_form", "analytic")),
    ("bivariate_rotation_sweep", (
        "bivariate_family_matrix", "rotation_cells", "r90", "r270",
    )),
    ("boundary", ("boundary", "extreme", "near_singular", "invalid_given")),
    ("conditional_pit", (
        "conditional_prediction_passes", "directional_pit", "mixture_pit",
        "uniform_reference_pit",
    )),
    ("dag_mcmc", ("dag_mcmc", "mcmc")),
    ("dynamic_conditioning", ("dynamic_conditioning", "given_only")),
    ("dynamic_parameter_path", (
        "dynamic_parameter_path", "predictive_parameter_path",
        "rowwise_dynamic",
    )),
    ("external_reference", ("pyvinecopulib", "copulae")),
    ("factor", ("factor",)),
    ("full_history_oracle", (
        "reference_filter", "full_history", "same_last_observation",
        "predictive_mixture_pit", "scaroureference", "mixture_pit",
        "pi_t_given_t", "pi_t_plus_1_given_t", "scar_tm_ou_oracle",
    )),
    ("high_dimension", ("high_dimension", "large_dimension", "d50", "d=50")),
    ("parallelism", ("n_threads", "parallel", "thread")),
    ("persistence", ("persistence", "save_load", "roundtrip")),
    ("reproducibility", ("reproduc", "fixed_rng", "seed")),
    ("suffix", ("suffix", "peel")),
)


@dataclass(frozen=True)
class TestFunction:
    """One source-level pytest function and its audit text."""

    node_base: str
    relative_file: str
    qualified_name: str
    source: str


class _TestVisitor(ast.NodeVisitor):
    def __init__(self, relative_file: str, lines: Sequence[str]) -> None:
        self.relative_file = relative_file
        self.lines = lines
        self.class_stack: list[str] = []
        self.tests: list[TestFunction] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _record_test(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not node.name.startswith("test_"):
            return
        decorator_lines = [
            decorator.lineno for decorator in node.decorator_list
            if hasattr(decorator, "lineno")
        ]
        start = min([node.lineno, *decorator_lines]) - 1
        end = int(node.end_lineno or node.lineno)
        source = "".join(self.lines[start:end])
        components = [self.relative_file, *self.class_stack, node.name]
        node_base = "::".join(components)
        qualified = ".".join([*self.class_stack, node.name])
        self.tests.append(TestFunction(
            node_base=node_base,
            relative_file=self.relative_file,
            qualified_name=qualified,
            source=source,
        ))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_test(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_test(node)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def discover_test_functions(
        include_conditional_contracts: bool) -> tuple[TestFunction, ...]:
    discovered: list[TestFunction] = []
    for path in sorted(TESTS.rglob("test_*.py")):
        relative = _relative(path)
        if (
                not include_conditional_contracts
                and relative.startswith("tests/conditional/")):
            continue
        # Some historical test modules carry an UTF-8 BOM.
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        visitor = _TestVisitor(relative, source.splitlines(keepends=True))
        visitor.visit(tree)
        discovered.extend(visitor.tests)
    return tuple(discovered)


def _strip_parameter_id(node_id: str) -> str:
    return re.sub(r"\[.*\]$", "", node_id).replace("\\", "/")


def collect_pytest_node_ids(
        include_conditional_contracts: bool) -> tuple[str, ...]:
    artifacts = ROOT / "benchmark_artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="conditional-collect-", dir=artifacts) as temp_dir:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--basetemp",
            str(Path(temp_dir) / "pytest"),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "pytest collection failed:\n" + completed.stdout + completed.stderr
        )
    node_ids = tuple(
        line.strip().replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    )
    if include_conditional_contracts:
        return node_ids
    return tuple(
        node_id for node_id in node_ids
        if not node_id.startswith("tests/conditional/")
    )


def _tags(text: str, patterns: Iterable[tuple[str, Sequence[str]]]) -> list[str]:
    lowered = text.lower()
    return [
        tag for tag, needles in patterns
        if any(needle in lowered for needle in needles)
    ]


def _regex_tags(
        text: str,
        patterns: Iterable[tuple[str, Sequence[str]]]) -> list[str]:
    return [
        tag for tag, expressions in patterns
        if any(re.search(expression, text, re.IGNORECASE)
               for expression in expressions)
    ]


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = completed.stdout.strip()
    return value or None


def build_inventory(
        include_conditional_contracts: bool = False) -> dict[str, object]:
    registry_bytes = REGISTRY_PATH.read_bytes()
    registry = json.loads(registry_bytes.decode("utf-8"))
    tests = discover_test_functions(
        include_conditional_contracts=include_conditional_contracts)
    node_ids = collect_pytest_node_ids(
        include_conditional_contracts=include_conditional_contracts)

    by_base = {test.node_base: test for test in tests}
    name_matched_functions = {
        test.node_base: test for test in tests
        if _CONDITIONAL_NAME_PATTERN.search(test.qualified_name)
    }
    conditional_functions = {
        test.node_base: test for test in tests
        if (
            _CONDITIONAL_NAME_PATTERN.search(test.qualified_name)
            or _CONDITIONAL_SOURCE_PATTERN.search(test.source)
            or (
                include_conditional_contracts
                and test.relative_file.startswith("tests/conditional/")
            )
        )
    }
    conditional_nodes: list[dict[str, object]] = []
    for node_id in node_ids:
        base = _strip_parameter_id(node_id)
        test = conditional_functions.get(base)
        if test is None:
            continue
        audit_text = node_id + "\n" + test.source
        models = _regex_tags(audit_text, _MODEL_PATTERNS)
        if not models and test.relative_file.startswith("tests/test_rvine"):
            models = ["vine-generic"]
        if not models and test.relative_file == "tests/test_vine.py":
            models = ["vine-legacy-cvine"]
        conditional_nodes.append({
            "node_id": node_id,
            "source_test": base,
            "models": models,
            "methods": _tags(audit_text, _METHOD_PATTERNS),
            "features": _tags(audit_text, _FEATURE_PATTERNS),
        })

    file_counts = Counter(
        str(item["node_id"]).split("::", 1)[0]
        for item in conditional_nodes
    )
    model_counts = Counter(
        model
        for item in conditional_nodes
        for model in item["models"]  # type: ignore[index]
    )
    method_counts = Counter(
        method
        for item in conditional_nodes
        for method in item["methods"]  # type: ignore[index]
    )
    feature_counts = Counter(
        feature
        for item in conditional_nodes
        for feature in item["features"]  # type: ignore[index]
    )
    registry_model_ids = [case["id"] for case in registry["model_cases"]]

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "registry": {
            "path": _relative(REGISTRY_PATH),
            "schema_version": registry["schema_version"],
            "sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "model_count": len(registry["model_cases"]),
            "conditional_case_count": sum(
                len(case["methods"])
                for case in registry["model_cases"]
            ),
            "unsupported_case_count": len(registry["unsupported_cases"]),
        },
        "collection": {
            "includes_conditional_contracts": include_conditional_contracts,
            "collected_case_count": len(node_ids),
            "source_test_function_count": len(tests),
            "conditional_name_function_count": len(name_matched_functions),
            "conditional_source_function_count": len(conditional_functions),
            "conditional_collected_case_count": len(conditional_nodes),
        },
        "inferred_coverage": {
            "by_file": dict(sorted(file_counts.items())),
            "by_model": {
                model_id: model_counts.get(model_id, 0)
                for model_id in registry_model_ids
            },
            "by_method": {
                method: method_counts.get(method, 0)
                for method, _ in _METHOD_PATTERNS
            },
            "by_feature": {
                feature: feature_counts.get(feature, 0)
                for feature, _ in _FEATURE_PATTERNS
            },
            "models_without_inferred_cases": [
                model_id for model_id in registry_model_ids
                if model_counts.get(model_id, 0) == 0
            ],
        },
        "conditional_cases": conditional_nodes,
        "limitations": [
            "The inventory is based on AST/name heuristics, not runtime coverage.",
            "Model tags may be absent when fixtures hide the concrete class.",
            "A collected parameter case can carry multiple model/method tags.",
            "By default tests/conditional is excluded for a "
            "separate-suite comparison."
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "JSON destination (default: "
            "benchmark_artifacts/conditional_sampling_inventory.json)"
        ),
    )
    parser.add_argument(
        "--include-conditional-contracts",
        dest="include_conditional_contracts",
        action="store_true",
        help="include tests/conditional in the inventory",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="also print the complete JSON document",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inventory = build_inventory(
        include_conditional_contracts=args.include_conditional_contracts)
    rendered = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    summary = inventory["collection"]
    print(
        f"wrote {output}: "
        f"{summary['conditional_source_function_count']} conditional test "
        f"functions, {summary['conditional_collected_case_count']} collected cases"
    )
    if args.stdout:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
