from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Type

from zero_copy_pipe.dto import BufferSpec

if TYPE_CHECKING:
    import torch


class IGpuProgram(AbstractContextManager, ABC):
    """Contract for a swappable GPU (or CPU) processing stage with pre-allocated buffers."""

    @property
    @abstractmethod
    def input_spec(self) -> BufferSpec:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_spec(self) -> BufferSpec:
        raise NotImplementedError

    @property
    @abstractmethod
    def uses_cuda(self) -> bool:
        raise NotImplementedError

    @property
    def num_streams(self) -> int:
        return 1

    @abstractmethod
    def setup(self) -> None:
        """Initialize device resources, model/kernels, and pre-allocated buffers."""
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        """Run a warm-up pass for every stream slot after setup()."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Synchronize and release device buffers cleanly. Must be idempotent."""
        raise NotImplementedError

    @abstractmethod
    def enqueue(
        self,
        host_input: torch.Tensor,
        host_output: torch.Tensor,
        stream_slot: int = 0,
    ) -> tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]:
        """Enqueue H2D -> compute -> D2H on `stream_slot`.

        Returns `(h2d_done_event, d2h_done_event)`, either or both of which may be
        `None` for synchronous/CPU implementations.
        """
        raise NotImplementedError

    def run(self, host_input: torch.Tensor, host_output: torch.Tensor) -> None:
        """Execute synchronously on stream slot 0."""
        _, done_ev = self.enqueue(host_input, host_output, stream_slot=0)
        if done_ev is not None:
            done_ev.synchronize()

    def __enter__(self) -> "IGpuProgram":
        self.setup()
        try:
            self.warmup()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()
