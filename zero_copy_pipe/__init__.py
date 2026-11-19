from typing import Any

from .dto import BufferSpec, PacketDto
from .interfaces import (
    CallableOutputProcessor,
    CallableOutputProcessorFactory,
    IGpuProgram,
    IOutputProcessor,
)
from .pipeline import (
    ActiveGroupTracker,
    InputSlot,
    PipeManager,
    PipelineClosedError,
    PipelineError,
    PipelineWorkerError,
    SharedMemoryArena,
)
from .programs import DebayerConfig, DebayerProgramFactory

__all__ = [
    "ActiveGroupTracker",
    "BufferSpec",
    "CallableOutputProcessor",
    "CallableOutputProcessorFactory",
    "Debayer5x5",
    "DebayerConfig",
    "DebayerProgram",
    "DebayerProgramFactory",
    "IGpuProgram",
    "IOutputProcessor",
    "InputSlot",
    "Layout",
    "PacketDto",
    "PipeManager",
    "PipelineClosedError",
    "PipelineError",
    "PipelineWorkerError",
    "SharedMemoryArena",
]


def __getattr__(name: str) -> Any:
    if name in {"Debayer5x5", "DebayerProgram", "Layout"}:
        from . import programs

        return getattr(programs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
