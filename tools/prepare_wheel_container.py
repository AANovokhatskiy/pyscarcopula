"""Keep cibuildwheel's fixed Linux container output paths below /project/build."""

from pathlib import Path
import sys


def prepare(project: Path, container_root: Path = Path("/")):
    generated = project.resolve() / "build" / "ci"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "tmp").mkdir(exist_ok=True)
    # cibuildwheel 3.4.1 uses these paths independently of TMPDIR.
    for relative, name, directory in (
        ("tmp/cibuildwheel", "cibuildwheel", True),
        ("output", "wheel-output", True),
        ("constraints.txt", "constraints.txt", False),
    ):
        target = generated / name
        if directory:
            target.mkdir(exist_ok=True)
        else:
            target.touch(exist_ok=True)
        link = container_root / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"refusing to replace an existing container path: {link}")
        link.symlink_to(target, target_is_directory=directory)


if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("this helper is only for the isolated Linux wheel container")
    prepare(Path(__file__).resolve().parents[1])
