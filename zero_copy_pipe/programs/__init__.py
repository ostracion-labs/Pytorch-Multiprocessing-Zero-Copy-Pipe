from typing import Any

from .debayer_config import DebayerConfig, DebayerProgramFactory

__all__ = [
    "Debayer5x5",
    "DebayerConfig",
    "DebayerProgram",
    "DebayerProgramFactory",
    "Layout",
]


def __getattr__(name: str) -> Any:
    if name in {"Debayer5x5", "Layout"}:
        from . import debayer_5x5

        return getattr(debayer_5x5, name)
    if name == "DebayerProgram":
        from . import debayer_program

        return debayer_program.DebayerProgram
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
