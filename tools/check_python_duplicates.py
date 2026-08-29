"""Detect exact Python function-body clones in shipped package modules."""

from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
from pathlib import Path


def exact_clone_groups(root: Path, *, minimum_nodes: int = 14) -> list[dict]:
    package = root.resolve() / "pyscarcopula"
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(root.resolve()).as_posix()
        tree = ast.parse(
            path.read_text(encoding="utf-8-sig"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = list(node.body)
            if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            node_count = sum(
                1 for statement in body for _ in ast.walk(statement))
            if node_count < minimum_nodes:
                continue
            normalized = ast.dump(
                ast.Module(body=body, type_ignores=[]),
                include_attributes=False,
            )
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            grouped[digest].append({
                "path": relative,
                "name": node.name,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "ast_nodes": node_count,
            })
    groups = [
        {"digest": digest, "members": members}
        for digest, members in grouped.items()
        if len(members) > 1
    ]
    return sorted(
        groups,
        key=lambda group: (
            -group["members"][0]["ast_nodes"],
            group["members"][0]["path"],
            group["members"][0]["line"],
        ),
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    groups = exact_clone_groups(root)
    for group in groups:
        members = ", ".join(
            f'{item["path"]}:{item["line"]} {item["name"]}'
            for item in group["members"])
        print(f'{group["members"][0]["ast_nodes"]} nodes: {members}')
    raise SystemExit(bool(groups))
