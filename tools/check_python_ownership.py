"""Conservative, full-package Python numerical ownership and import gate.

No production imports are executed. Every function, lambda, class body and
module body is inventoried, including code not reached by public imports.
Arithmetic and numerical calls require an exact reviewed symbol exception.
Import reachability is a conservative module graph, NOT a call-graph proof
that a function executes. Wheel-importable code is never exempted as dead.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, deque
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

try:
    from tools.python_ownership_policy import EXCEPTIONS
except ModuleNotFoundError:
    from python_ownership_policy import EXCEPTIONS


RAW = "pyscarcopula._native._scar_cpp"
REMOVED_RAW = "pyscarcopula._scar_cpp"
NUMERIC_CALLS = frozenset("""
exp expm1 exp2 log log1p log2 log10 sqrt cbrt power square reciprocal
sin cos tan arcsin arccos arctan arctan2 sinh cosh tanh arcsinh arccosh arctanh
sum nansum prod cumprod cumsum mean nanmean var nanvar std nanstd median
quantile percentile histogram histogram2d interp gradient diff dot inner outer min max
einsum tensordot matmul cov corrcoef trapz trapezoid linspace logspace geomspace
clip maximum minimum fmax fmin abs fabs sign sinpi cospi
logaddexp logaddexp2 round around floor ceil rint ptp
""".split())
BUILTIN_NUMERIC = frozenset({"sum", "abs", "pow", "round", "min", "max"})
RNG_CALLS = frozenset({
    "default_rng", "random", "uniform", "normal", "standard_normal",
    "integers", "choice", "permutation", "shuffle", "spawn", "SeedSequence",
})
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _stable_ast_dump(value):
    """Return the Python 3.12 AST representation on every supported CPython.

    Python 3.13 changed :func:`ast.dump` to omit empty fields by default,
    while Python 3.10 and 3.11 do not expose the ``type_params`` fields added
    to definitions in Python 3.12.  Ownership review fingerprints must describe
    source structure, not the interpreter used to run the release check.
    """
    if isinstance(value, ast.AST):
        fields = list(ast.iter_fields(value))
        definition_types = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        )
        if isinstance(value, definition_types) and not any(
                name == "type_params" for name, _ in fields):
            fields.append(("type_params", getattr(value, "type_params", [])))
        fields = [
            (name, field_value)
            for name, field_value in fields
            if not (
                field_value is None
                and getattr(type(value), name, ...) is None
            )
        ]
        rendered = ", ".join(
            f"{name}={_stable_ast_dump(field_value)}"
            for name, field_value in fields
        )
        return f"{type(value).__name__}({rendered})"
    if isinstance(value, list):
        return "[" + ", ".join(_stable_ast_dump(item) for item in value) + "]"
    return repr(value)


def fingerprint(node):
    canonical = _stable_ast_dump(node)
    return hashlib.sha256(canonical.encode()).hexdigest()


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return dotted(node.func) + "()"
    return ""


def resolved(name, aliases):
    parts = name.split(".")
    for index in range(len(parts), 0, -1):
        prefix = ".".join(parts[:index])
        if prefix in aliases:
            return ".".join([aliases[prefix], *parts[index:]])
    return name


def module_name(path, root):
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def import_base(node, module, is_package):
    if not node.level:
        return node.module or ""
    package = module if is_package else module.rpartition(".")[0]
    try:
        return importlib.util.resolve_name("." * node.level + (node.module or ""), package)
    except (ValueError, ImportError):
        return "<unresolved-relative-import>"


def _type_expression(node):
    """Recognize declarative types without exempting executable expressions."""
    if isinstance(node, (ast.Name, ast.Attribute)):
        return bool(dotted(node))
    if isinstance(node, ast.Constant):
        return node.value is None or isinstance(node.value, str)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _type_expression(node.left) and _type_expression(node.right)
    if isinstance(node, ast.Subscript):
        return _type_expression(node.value) and _type_expression(node.slice)
    if isinstance(node, ast.Tuple):
        return all(_type_expression(item) for item in node.elts)
    return False


def own_nodes(node):
    """Include defaults/decorators, exclude annotations and nested bodies."""
    def visit(item, initial=False):
        if isinstance(item, SCOPES) and not initial:
            return
        yield item
        for field, value in ast.iter_fields(item):
            if field in {"returns", "annotation", "type_comment", "type_params"}:
                continue
            if (isinstance(item, ast.AnnAssign) and field == "value"
                    and dotted(item.annotation) in {"TypeAlias", "typing.TypeAlias"}
                    and _type_expression(value)):
                continue
            if isinstance(value, list):
                for child in value:
                    if isinstance(child, ast.AST):
                        yield from visit(child)
            elif isinstance(value, ast.AST):
                yield from visit(value)
    yield from visit(node, True)


def scopes(tree):
    yield "<module>", tree
    counts = Counter()
    definitions = Counter()

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, SCOPES):
                if isinstance(child, ast.Lambda):
                    counts[prefix] += 1
                    name = f"<lambda#{counts[prefix]}>"
                else:
                    name = child.name
                    definitions[(prefix, name)] += 1
                    if definitions[(prefix, name)] > 1:
                        name += f"#{definitions[(prefix, name)]}"
                key = f"{prefix}.{name}" if prefix else name
                yield key, child
                yield from walk(child, key)
            else:
                yield from walk(child, prefix)
    yield from walk(tree, "")


def collect_imports(tree, module, is_package):
    aliases, imports = {}, []
    # All conditional/lazy imports are included. An import in a dead branch
    # can overestimate reachability but cannot hide a numerical implementation.
    local = set(own_nodes(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if node in local:
                    aliases[item.asname or item.name.split(".")[0]] = (
                        item.name if item.asname else item.name.split(".")[0])
                imports.append((item.name, node.lineno, "import"))
        elif isinstance(node, ast.ImportFrom):
            base = import_base(node, module, is_package)
            for item in node.names:
                target = f"{base}.{item.name}"
                if node in local:
                    aliases[item.asname or item.name] = target
                imports.append((target, node.lineno, "from"))
    # Resolve simple assignment aliases, e.g. quantile = special.stdtrit.
    for _ in range(3):
        for node in local:
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Name, ast.Attribute)):
                value = resolved(dotted(node.value), aliases)
                for target in node.targets:
                    if isinstance(target, ast.Name) and value:
                        aliases[target.id] = value
    return aliases, imports


def literal_dynamic_import(node, aliases, module, is_package):
    if not isinstance(node, ast.Call):
        return None
    name = resolved(dotted(node.func), aliases)
    if name not in {"importlib.import_module", "__import__"}:
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        return "<dynamic-import>", node.lineno, "dynamic"
    value = node.args[0].value
    if value.startswith("."):
        package = module if is_package else module.rpartition(".")[0]
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            package = node.args[1].value
        try:
            value = importlib.util.resolve_name(value, package)
        except (ValueError, ImportError):
            value = "<dynamic-import>"
    return value, node.lineno, "dynamic"


def signals(node, aliases, module="", is_package=False):
    nodes = list(own_nodes(node))
    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    signals_out, calls = [], []

    def add(item, rule, detail):
        signals_out.append(dict(rule=rule, line=getattr(item, "lineno", 1),
            detail=detail, expression=(getattr(item, "name", "<lambda>")
                if isinstance(item, SCOPES) else ast.unparse(item)),
            fingerprint=fingerprint(item)))

    def ancestor(item, node_type):
        current = item
        while current in parents:
            current = parents[current]
            if isinstance(current, node_type):
                return current
        return None

    def fallback_value(value):
        if value is None:
            return False
        policy_value = value
        while isinstance(policy_value, ast.Subscript):
            policy_value = policy_value.value
        if isinstance(policy_value, ast.Call):
            call_name = resolved(dotted(policy_value.func), aliases)
            if call_name.startswith("pyscarcopula._native.model_policy."):
                return False
        for child in ast.walk(value):
            if isinstance(child, ast.Name) and re.search(
                    r"(^|_)(fail|failure|penalty)(_|$)", child.id.lower()):
                return True
            if isinstance(child, ast.Attribute) and re.search(
                    r"(^|_)(fail|failure|penalty)(_|$)", child.attr.lower()):
                return True
            if isinstance(child, ast.Constant) and isinstance(
                    child.value, (int, float)) and not isinstance(child.value, bool):
                if abs(float(child.value)) >= 1e6:
                    return True
            if isinstance(child, ast.Call):
                leaf = resolved(dotted(child.func), aliases).rsplit(".", 1)[-1]
                if leaf in {
                        "zeros", "ones", "zeros_like", "ones_like",
                        "full", "full_like"}:
                    return True
        return False

    def direct_failure_reference(value):
        if isinstance(value, ast.Name):
            return bool(re.search(
                r"(^|_)(fail|failure|penalty)(_|$)", value.id.lower()))
        if isinstance(value, ast.Attribute):
            return bool(re.search(
                r"(^|_)(fail|failure|penalty)(_|$)", value.attr.lower()))
        if isinstance(value, (ast.Tuple, ast.List)):
            return any(direct_failure_reference(element) for element in value.elts)
        if isinstance(value, ast.Call):
            leaf = resolved(dotted(value.func), aliases).rsplit(".", 1)[-1]
            return leaf in {"float", "int"} and any(
                direct_failure_reference(argument) for argument in value.args)
        return False

    def numerical_guard(item):
        guard = ancestor(item, ast.If)
        if guard is None:
            return False
        names = " ".join(
            dotted(child).lower() for child in ast.walk(guard.test)
            if isinstance(child, (ast.Name, ast.Attribute, ast.Call)))
        return any(marker in names for marker in (
            "status", "finite", "valid", "invalid", "failure",
            "support", "domain"))

    for item in nodes:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            names = [dotted(target).lower() for target in targets]
            if any(re.search(r"(bounds|_df_offset|_ou_.*(lower|upper)|_adaptive_.*order)", name)
                   for name in names):
                if item.value is not None and any(
                        isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                        and not isinstance(n.value, bool) for n in ast.walk(item.value)):
                    add(item, "model-policy", "Python-owned literal model bounds/defaults")
        elif isinstance(item, ast.BinOp):
            # Type unions/annotations were excluded above. Index arithmetic
            # and literal string/sequence composition are structural.
            current = item
            index_expression = False
            while current in parents:
                parent = parents[current]
                if isinstance(parent, ast.Subscript) and parent.slice is current:
                    index_expression = True
                    break
                current = parent
            sequence = (ast.Tuple, ast.List, ast.Set, ast.JoinedStr)
            structural = isinstance(item.left, sequence) or isinstance(item.right, sequence)
            structural |= any(isinstance(x, ast.Constant) and isinstance(x.value, (str, bytes))
                              for x in (item.left, item.right))
            if not index_expression and not structural:
                add(item, "arithmetic", type(item.op).__name__)
        elif isinstance(item, ast.AugAssign):
            add(item, "arithmetic", type(item.op).__name__ + "=")
        elif isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.USub, ast.UAdd)):
            if not isinstance(item.operand, ast.Constant):
                add(item, "arithmetic", type(item.op).__name__)
        elif isinstance(item, ast.Call):
            name = resolved(dotted(item.func), aliases)
            calls.append(dict(name=name, line=item.lineno))
            leaf = name.rsplit(".", 1)[-1]
            if name.startswith(("numpy.linalg.", "scipy.linalg.", "scipy.special.", "scipy.stats.")):
                add(item, "numerical-call", name)
            elif name.startswith(("math.", "cmath.")):
                add(item, "numerical-call", name)
            elif name in BUILTIN_NUMERIC or (
                    leaf in NUMERIC_CALLS and not name.startswith("pyscarcopula._native.")):
                add(item, "numerical-call", name)
            if name.startswith("numba.") and leaf in {"njit", "jit", "vectorize", "guvectorize", "stencil"}:
                add(item, "numba-kernel", name)
            if name in {"eval", "exec", "compile"}:
                add(item, "opaque-execution", name)
            if any(k.arg == "initial_parameters" and any(
                    isinstance(n, ast.Constant) and isinstance(n.value, float)
                    for n in ast.walk(k.value)) for k in item.keywords):
                add(item, "model-policy", "literal model initial parameters passed to optimizer")
            if name.endswith((".transform", ".inv_transform")) and any(
                    isinstance(n, ast.Constant) and isinstance(n.value, float)
                    for argument in item.args for n in ast.walk(argument)):
                add(item, "model-policy", "literal model initialization through parameter transform")
            dynamic = literal_dynamic_import(item, aliases, module, is_package)
            if dynamic and dynamic[0] == "<dynamic-import>":
                add(item, "dynamic-import", "Nonliteral import; possible targets require explicit review")
            # Nonstandard model-distribution draws cross the raw-RNG boundary.
            if leaf in {"beta", "gamma", "chisquare", "standard_t", "multivariate_normal"}:
                add(item, "model-rng", name)
            elif leaf == "choice" and any(k.arg == "p" for k in item.keywords):
                add(item, "model-rng", "weighted state sampling: " + name)
            elif leaf in {"uniform", "normal"}:
                params = list(item.args[:2]) + [
                    k.value for k in item.keywords if k.arg in {"low","high","loc","scale"}]
                if any(not isinstance(p, ast.Constant) for p in params):
                    add(item, "model-rng", "parameterized draws: " + name)
            # Constant penalties/fallback arrays do not contain arithmetic.
            if any(isinstance(p, ast.ExceptHandler) and item in set(ast.walk(p)) for p in nodes):
                if leaf in {"zeros", "ones", "zeros_like", "ones_like", "full", "full_like"}:
                    add(item, "numeric-fallback", name)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in item.decorator_list:
                if isinstance(decorator, (ast.Name, ast.Attribute)):
                    name = resolved(dotted(decorator), aliases)
                    if name.startswith("numba."):
                        add(decorator, "numba-kernel", name)
        elif isinstance(item, ast.ImportFrom) and any(a.name == "*" for a in item.names):
            add(item, "opaque-import", "wildcard import requires review")
        elif isinstance(item, (ast.Import, ast.ImportFrom)):
            if isinstance(item, ast.ImportFrom):
                base = import_base(item, module, is_package)
                imported_names = [base + "." + a.name for a in item.names]
            else:
                imported_names = [a.name for a in item.names]
            for name in imported_names:
                if name.startswith(("scipy.special", "scipy.linalg", "scipy.stats", "numpy.linalg", "numba")):
                    add(item, "numerical-import", name)
        elif isinstance(item, ast.Return) and fallback_value(item.value):
            if ancestor(item, ast.ExceptHandler) is not None:
                add(item, "numeric-fallback",
                    "synthetic numerical result in exception handler")
            elif numerical_guard(item):
                add(item, "numeric-fallback",
                    "synthetic numerical result behind status/domain guard")
            elif direct_failure_reference(item.value):
                add(item, "numeric-fallback",
                    "synthetic failure/penalty result returned by Python")
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Literal model defaults/bounds need an owner even without a formula.
        model_policy = bool(re.search(r"(^|_)(bounds|initial_point|automatic_gas_start)$", node.name))
        model_policy |= node.name in {"ou_params", "jacobi_params", "gas_params"}
        if model_policy and any(isinstance(n, ast.Constant) and isinstance(n.value, float) for n in nodes):
            add(node, "model-policy", "numerical initialization/domain/default policy")
        if "adaptive" in node.name and "basis" in node.name and any(isinstance(n, ast.Compare) for n in nodes):
            add(node, "model-policy", "parameter-dependent approximation order")
    if module.startswith("pyscarcopula.vine"):
        for loop in [n for n in nodes if isinstance(n, (ast.For, ast.While))]:
            # Structure-search/fit orchestration can be reviewed separately.
            for call in (n for n in ast.walk(loop) if isinstance(n, ast.Call)):
                name = resolved(dotted(call.func), aliases)
                if name.rsplit(".", 1)[-1] in {"h_pair", "h_inverse", "_edge_h_pair", "_edge_h_pair_for_variables", "advance_forward_phi"}:
                    add(call, "numerical-traversal", name)
    return signals_out, calls


def paths_from_roots(graph, roots):
    paths = {root: [root] for root in roots if root in graph}
    pending = deque(paths)
    while pending:
        parent = pending.popleft()
        for target in sorted(graph[parent]):
            if target not in paths:
                paths[target] = paths[parent] + [target]
                pending.append(target)
    return paths


def audit_package(root, exceptions=None):
    root = Path(root).resolve()
    package = root / "pyscarcopula"
    policy = EXCEPTIONS if exceptions is None else exceptions
    files = sorted(package.rglob("*.py"))
    files = [p for p in files if "_cpp" not in p.relative_to(package).parts]
    modules = {module_name(p, root): p for p in files}
    graph = {m: set() for m in modules}
    entries, edges, violations, used = [], [], [], set()
    if not files:
        violations.append(dict(rule="python-ownership-empty-package", path="pyscarcopula",
                               line=1, symbol="<package>", message="No production Python files found"))

    for module, path in modules.items():
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        except (OSError, SyntaxError) as error:
            violations.append(dict(rule="python-ownership-parse", path=relative,
                line=getattr(error, "lineno", 1), symbol="<module>", message=str(error)))
            continue
        aliases, imports = collect_imports(tree, module, path.name == "__init__.py")
        for node in ast.walk(tree):
            dynamic = literal_dynamic_import(node, aliases, module, path.name == "__init__.py")
            if dynamic:
                imports.append(dynamic)
        for imported, line, kind in imports:
            target = imported
            while target and target not in modules:
                target = target.rpartition(".")[0]
            if target:
                graph[module].add(target)
                # Importing a submodule also executes each package initializer.
                parts = target.split(".")
                for i in range(1, len(parts)):
                    parent = ".".join(parts[:i])
                    if parent in modules and modules[parent].name == "__init__.py":
                        graph[module].add(parent)
            edges.append(dict(source=module, target=target or None,
                imported=imported, line=line, kind=kind))
            raw = any(imported == prefix or imported.startswith(prefix + ".")
                      for prefix in (RAW, REMOVED_RAW))
            if raw and not (module == "pyscarcopula._native._extension" and imported == RAW):
                violations.append(dict(rule="python-ownership-raw-import", path=relative,
                    line=line, symbol="<module>", message=f"Raw extension import: {imported}"))
            if target.startswith("pyscarcopula.contrib") and not module.startswith("pyscarcopula.contrib"):
                violations.append(dict(rule="python-ownership-contrib-boundary",path=relative,
                    line=line,symbol="<module>",message=f"Copula core imports out-of-scope {target}"))
            if imported.startswith("pyscarcopula._cpp"):
                violations.append(dict(rule="python-ownership-build-helper-import",path=relative,
                    line=line,symbol="<module>",message="Production imports excluded C++ build-support Python"))

        scope_aliases = {"": aliases}
        for symbol, node in scopes(tree):
            key = module + ":" + symbol
            parent = symbol.rpartition(".")[0]
            local_aliases, _ = collect_imports(node, module, path.name == "__init__.py")
            local_aliases = {**scope_aliases.get(parent, aliases), **local_aliases}
            scope_aliases[symbol] = local_aliases
            candidates, calls = signals(node, local_aliases, module, path.name == "__init__.py")
            for item in own_nodes(node):
                dynamic = literal_dynamic_import(item, local_aliases, module, path.name == "__init__.py")
                if dynamic and any(dynamic[0] == prefix or dynamic[0].startswith(prefix + ".")
                                   for prefix in (RAW, REMOVED_RAW)):
                    if not (module == "pyscarcopula._native._extension" and dynamic[0] == RAW):
                        violations.append(dict(rule="python-ownership-raw-import", path=relative,
                            line=dynamic[1],symbol=symbol,message=f"Raw dynamic extension import: {dynamic[0]}"))
            exception = policy.get(key)
            allowed = []
            remaining = candidates[:]
            if exception:
                used.add(key)
                valid = all(exception.get(f) for f in ("owner", "reason", "test", "fingerprint", "category"))
                valid &= exception.get("fingerprint") == fingerprint(node)
                if valid:
                    allowed = [s for s in candidates if s["rule"] in exception.get("rules", ())]
                    remaining = [s for s in candidates if s not in allowed]
                else:
                    violations.append(dict(rule="python-ownership-stale-allowlist", path=relative,
                        line=getattr(node,"lineno",1),symbol=symbol,
                        message=f"Missing review metadata or changed AST for {key}"))
            excluded = module.startswith("pyscarcopula.contrib")
            if excluded:
                category = "out_of_scope"
            elif any(s["rule"] not in {"dynamic-import", "opaque-import", "opaque-execution"} for s in remaining):
                category = "model_math"
            elif exception:
                category = exception["category"]
            elif any(c["name"].startswith("scipy.optimize.") or c["name"].endswith((".fit",".minimize"))
                     for c in calls):
                category = "fit"
            elif any(c["name"].rsplit(".",1)[-1] in RNG_CALLS for c in calls):
                category = "rng"
            elif any(c["name"].startswith("pyscarcopula._native.") for c in calls) or not calls:
                category = "adapter/DTO/validation"
            else:
                category = "orchestration"
            if not excluded:
                for candidate in remaining:
                    violations.append(dict(rule="python-ownership-" + candidate["rule"],
                        path=relative,line=candidate["line"],symbol=symbol,
                        message=candidate["detail"]))
            entries.append(dict(key=key,module=module,path=relative,symbol=symbol,
                kind=type(node).__name__,line=getattr(node,"lineno",1),
                end_line=getattr(node,"end_lineno",len(path.read_text(encoding="utf-8-sig").splitlines())),
                fingerprint=fingerprint(node),category=category,signals=candidates,
                allowed_signals=allowed,unreviewed_signals=remaining,
                calls=calls,exception=exception,scope="contrib" if excluded else "copula_core"))
    roots = ["pyscarcopula","pyscarcopula.api","pyscarcopula.io","pyscarcopula.stattests"]
    roots += [m for m in modules if m.startswith(("pyscarcopula.strategy.", "pyscarcopula._native."))]
    paths = paths_from_roots(graph, roots)
    for entry in entries:
        entry["import_path"] = paths.get(entry["module"])
        entry["reachability"] = "conservative_core_import_path" if entry["import_path"] else "direct_import_available"
    graph_data = dict(roots=roots,modules=[
        dict(module=m,path=p.relative_to(root).as_posix(),import_path=paths.get(m),
             wheel_importable=True,scope="contrib" if m.startswith("pyscarcopula.contrib") else "copula_core")
        for m,p in modules.items()], edges=edges,
        graph={m:sorted(targets) for m,targets in graph.items()},
        limitations="Module graph overapproximates conditional/lazy imports. It does not prove function execution. "
                    "No unreachable-compatibility exemptions; all shipped numerical code remains subject to the gate.",
        contrib_reachable_from_core=[m for m in paths if m.startswith("pyscarcopula.contrib")])
    # One AST site may be visible through both the module graph and its scope.
    violations = list({(v["rule"],v["path"],v["line"],v["symbol"],v["message"]):v
                       for v in violations}.values())
    return dict(schema_version=1,repository_root=str(root),files=len(files),entries=entries,
        categories=dict(Counter(e["category"] for e in entries)),
        functions=sum(e["kind"] in {"FunctionDef","AsyncFunctionDef","Lambda"} for e in entries),
        violations=violations,unused_exception_keys=sorted(set(policy)-used),
        import_graph=graph_data,verdict="FAIL" if violations else "PASS")


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root",type=Path)
    args=parser.parse_args(argv)
    root=args.root.resolve()
    if args.artifact_root:
        output=args.artifact_root.resolve()
        if output.is_relative_to(root) and not output.is_relative_to(root / "build"):
            parser.error("--artifact-root must be inside build/ or outside the product repository")
        targets=[output/n for n in ("python_inventory.json","python_import_graph.json","python_ownership_gate.json")]
        if any(p.exists() for p in targets) or (output/"checksums.sha256").exists():
            parser.error("Refusing to overwrite audit artifacts; use a new external directory")
    result=audit_package(root)
    if args.artifact_root:
        output.mkdir(parents=True,exist_ok=True)
        graph=result.pop("import_graph")
        summary={k:v for k,v in result.items() if k!="entries"}
        for path,data in zip(targets,(result,graph,summary)):
            path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Python ownership: {result['verdict']}; {result['files']} files, "
          f"{result['functions']} functions/lambdas, {len(result['violations'])} violations")
    for item in result["violations"]:
        print(f"[{item['rule']}] {item['path']}:{item['line']} {item['symbol']}: {item['message']}")
    return bool(result["violations"])


if __name__=="__main__":
    raise SystemExit(main())
