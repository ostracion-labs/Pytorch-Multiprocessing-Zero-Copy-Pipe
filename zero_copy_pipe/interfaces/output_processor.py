from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Optional, Type

import numpy as np

from zero_copy_pipe.dto import PacketDto


class IOutputProcessor(AbstractContextManager, ABC):
    """Contract for downstream CPU workers consuming read-only output shared memory views."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize worker-local state once at process startup."""
        raise NotImplementedError

    @abstractmethod
    def process(self, frame: np.ndarray, packet: PacketDto) -> Any:
        """Process one read-only output frame and optionally return a result."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release worker-local resources cleanly on shutdown."""
        raise NotImplementedError

    def __enter__(self) -> "IOutputProcessor":
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()


class CallableOutputProcessor(IOutputProcessor):
    """Adapts a plain callable `(frame, packet) -> Any` to `IOutputProcessor`."""

    def __init__(self, fn: Callable[[np.ndarray, PacketDto], Any]) -> None:
        self._fn = fn

    def setup(self) -> None:
        pass

    def process(self, frame: np.ndarray, packet: PacketDto) -> Any:
        return self._fn(frame, packet)

    def close(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class CallableOutputProcessorFactory:
    """Picklable factory wrapping a plain callable for spawned CPU workers."""

    fn: Callable[[np.ndarray, PacketDto], Any]

    def __call__(self) -> IOutputProcessor:
        return CallableOutputProcessor(self.fn)
