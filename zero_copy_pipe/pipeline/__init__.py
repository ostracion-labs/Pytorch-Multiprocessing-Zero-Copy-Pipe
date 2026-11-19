from typing import Any

from .active_group import ActiveGroupTracker
from .cpu_worker import DownstreamArenaView, cpu_worker_loop
from .errors import PipelineClosedError, PipelineError, PipelineWorkerError
from .manager import InputSlot, PipeManager
from .shm_arena import ArenaDescriptor, SharedMemoryArena

__all__ = [
    "ActiveGroupTracker",
    "ArenaDescriptor",
    "DownstreamArenaView",
    "InputSlot",
    "PipeManager",
    "PipelineClosedError",
    "PipelineError",
    "PipelineWorkerError",
    "SharedMemoryArena",
    "cpu_worker_loop",
    "gpu_worker_loop",
]


def __getattr__(name: str) -> Any:
    if name == "gpu_worker_loop":
        from . import gpu_worker

        return gpu_worker.gpu_worker_loop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
