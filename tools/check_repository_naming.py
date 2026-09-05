"""Check source and documentation names for numbered development milestones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SOURCE_DIRECTORIES = ("pyscarcopula", "tests", "docs", "tools", "benchmarks", ".github")
SOURCE_SUFFIXES = {".py", ".cpp", ".hpp", ".c", ".md", ".json", ".yml", ".yaml", ".ipynb"}
NUMBERED_MILESTONE = re.compile(
    r"(?<![a-z])(?:stage|gate|phase|fv|wp)[\s_-]*\d+|staged\s+migration",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NamingViolation:
    path: Path
    line: int
    match: str


def source_files(root: Path):
    """Include untracked/ignored source files but exclude generated build trees."""
    paths = {*root.glob("*.md"), root / ".gitignore", root / "MANIFEST.in"}
    for name in SOURCE_DIRECTORIES:
        for path in (root / name).rglob("*"):
            if (path.is_file() and path.suffix in SOURCE_SUFFIXES
                    and not {"__pycache__", "build", "site", ".git"}.intersection(
                        path.relative_to(root).parts)):
                paths.add(path)
    return sorted(path for path in paths if path.is_file())


def check_repository(root: Path) -> list[NamingViolation]:
    violations = []
    for path in source_files(root):
        relative = path.relative_to(root)
        if match := NUMBERED_MILESTONE.search(relative.as_posix()):
            violations.append(NamingViolation(relative, 0, match.group()))
        for number, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if match := NUMBERED_MILESTONE.search(text):
                violations.append(NamingViolation(relative, number, match.group()))
    return violations


if __name__ == "__main__":
    failures = check_repository(Path(__file__).resolve().parents[1])
    for failure in failures:
        print(f"{failure.path}:{failure.line}: numbered development label {failure.match!r}")
    raise SystemExit(bool(failures))
