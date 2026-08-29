"""Shared C++17 compiler policy for the extension and standalone tests."""

from __future__ import annotations

import os
import sys
from typing import Any


CXX_STANDARD = 17


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def requested_compiler() -> str | None:
    value = os.environ.get("PYSCA_CPP_COMPILER", "").strip()
    return value or None


def prepare_compiler_environment(compiler: Any) -> None:
    """Initialize MSVC and export its discovered toolchain ``PATH``.

    Windows can pass a process both ``Path`` and ``PATH`` entries.  Although
    environment names are case-insensitive there, ``cmd.exe`` preserves both,
    and the vcvars capture used by setuptools can select the stale entry.  A
    delete-and-restore collapses that malformed environment before MSVC is
    initialized.  Setuptools then keeps the vcvars path on the compiler rather
    than exporting it, so copy it back for tools spawned by ``link.exe`` (most
    notably ``rc.exe`` for embedded manifests).
    """

    if getattr(compiler, "compiler_type", None) != "msvc":
        return

    inherited_path = os.environ.get("PATH")
    if inherited_path is not None:
        del os.environ["PATH"]
        os.environ["PATH"] = inherited_path

    initialize = getattr(compiler, "initialize", None)
    if callable(initialize) and not getattr(compiler, "initialized", True):
        initialize()

    toolchain_path = getattr(compiler, "_paths", "")
    if toolchain_path:
        os.environ["PATH"] = toolchain_path

    if env_flag("PYSCA_CPP_STRICT"):
        # setuptools' MSVC defaults contain /W3.  Leaving it in place before
        # the explicit /W4 emits D9025, a command-line warning which /WX does
        # not promote.  Replace only the default warning level; /W4 and /WX
        # remain explicit per-target arguments below.
        for attribute in ("compile_options", "compile_options_debug"):
            options = getattr(compiler, attribute, None)
            if options is not None:
                setattr(
                    compiler,
                    attribute,
                    [
                        option for option in options
                        if str(option).upper() not in {
                            "/W0", "/W1", "/W2", "/W3", "/W4",
                        }
                    ],
                )


def _sanitizer_mode() -> tuple[bool, bool]:
    sanitize = env_flag("PYSCA_CPP_SANITIZE")
    thread_sanitize = env_flag("PYSCA_CPP_THREAD_SANITIZE")
    if sanitize and thread_sanitize:
        raise RuntimeError(
            "PYSCA_CPP_SANITIZE and PYSCA_CPP_THREAD_SANITIZE are mutually "
            "exclusive"
        )
    if (sanitize or thread_sanitize) and sys.platform == "win32":
        variable = (
            "PYSCA_CPP_SANITIZE" if sanitize
            else "PYSCA_CPP_THREAD_SANITIZE"
        )
        raise RuntimeError(
            f"{variable} requires a GCC- or Clang-compatible platform"
        )
    return sanitize, thread_sanitize


def extension_compile_args() -> list[str]:
    """Flags supplied to ``Pybind11Extension`` before compiler selection."""

    args: list[str] = []
    if env_flag("PYSCA_CPP_STRICT"):
        if sys.platform == "win32":
            args.extend(["/W4", "/WX"])
        else:
            args.extend(["-Wall", "-Wextra", "-Wpedantic", "-Werror"])

    sanitize, thread_sanitize = _sanitizer_mode()
    if sanitize:
        args.extend([
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
            "-O1",
            "-g",
        ])
    if thread_sanitize:
        args.extend([
            "-fsanitize=thread",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
            "-O1",
            "-g",
        ])
    return args


def extension_link_args() -> list[str]:
    sanitize, thread_sanitize = _sanitizer_mode()
    if sanitize:
        return [
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
        ]
    if thread_sanitize:
        return [
            "-fsanitize=thread",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
        ]
    return []


def standalone_compile_args(compiler_type: str) -> list[str]:
    """Return C++17 flags for setuptools' standalone compiler instance."""

    if compiler_type == "msvc":
        args = ["/std:c++17", "/EHsc", "/bigobj"]
        if env_flag("PYSCA_CPP_STRICT"):
            args.extend(["/W4", "/WX"])
        return args

    args = ["-std=c++17"]
    if env_flag("PYSCA_CPP_STRICT"):
        args.extend(["-Wall", "-Wextra", "-Wpedantic", "-Werror"])

    sanitize, thread_sanitize = _sanitizer_mode()
    if sanitize:
        args.extend([
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
            "-O1",
            "-g",
        ])
    if thread_sanitize:
        args.extend([
            "-fsanitize=thread",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
            "-O1",
            "-g",
        ])
    return args


def standalone_link_args(compiler_type: str) -> list[str]:
    if compiler_type == "msvc":
        return []
    return extension_link_args()


def prepare_mingw_extension(extension: Any) -> None:
    """Translate pybind/MSVC-style flags after setuptools selects MinGW."""

    translated = []
    for argument in extension.extra_compile_args:
        if argument.startswith("/std:"):
            translated.append("-std=" + argument[len("/std:"):])
        elif argument == "/W4":
            translated.extend(["-Wall", "-Wextra"])
        elif argument == "/WX":
            translated.append("-Werror")
        elif argument in ("/EHsc", "/bigobj"):
            continue
        else:
            translated.append(argument)
    if "-fvisibility=hidden" not in translated:
        translated.append("-fvisibility=hidden")
    extension.extra_compile_args = ["-O2", "-DNDEBUG", *translated]
    extension.extra_link_args = [
        *extension.extra_link_args,
        "-static-libstdc++",
        "-static-libgcc",
        # The Python import library precedes these flags in setuptools' link
        # command and remains dynamic.  Default GCC/MinGW runtime libraries,
        # which the driver appends later, are linked statically.
        "-static",
    ]
