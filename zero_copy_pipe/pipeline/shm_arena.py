from contextlib import AbstractContextManager
from dataclasses import dataclass
import logging
from multiprocessing.shared_memory import SharedMemory
from types import TracebackType
from typing import List, Optional, Type

import numpy as np

from zero_copy_pipe.dto import BufferSpec
from zero_copy_pipe.pipeline.errors import PipelineClosedError

logger = logging.getLogger(__name__)

PAGE_SIZE = 4096


def align_up(n: int, alignment: int = PAGE_SIZE) -> int:
    """Round `n` up to the nearest multiple of `alignment`."""
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return ((n + alignment - 1) // alignment) * alignment


@dataclass(frozen=True, slots=True)
class ArenaDescriptor:
    """Picklable descriptor for a contiguous, page-aligned SharedMemory arena."""

    name: str
    num_slots: int
    slot_stride: int
    total_bytes: int
    spec: BufferSpec

    def slot_offset(self, slot_id: int) -> int:
        if not (0 <= slot_id < self.num_slots):
            raise IndexError(f"slot_id {slot_id} out of range [0, {self.num_slots})")
        return slot_id * self.slot_stride


class SharedMemoryArena(AbstractContextManager):
    """Owns one contiguous, page-aligned POSIX SharedMemory arena partitioned into slots."""

    def __init__(self, num_slots: int, spec: BufferSpec) -> None:
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")

        slot_stride = align_up(spec.nbytes, PAGE_SIZE)
        total_bytes = num_slots * slot_stride

        shm = SharedMemory(create=True, size=total_bytes)
        views: List[np.ndarray] = []
        try:
            np_dtype = spec.numpy_dtype
            for idx in range(num_slots):
                views.append(
                    np.ndarray(
                        spec.shape,
                        dtype=np_dtype,
                        buffer=shm.buf,
                        offset=idx * slot_stride,
                    )
                )
        except BaseException:
            views.clear()
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            raise

        self._shm: Optional[SharedMemory] = shm
        self._views: List[np.ndarray] = views
        self._descriptor = ArenaDescriptor(
            name=shm.name,
            num_slots=num_slots,
            slot_stride=slot_stride,
            total_bytes=total_bytes,
            spec=spec,
        )
        self._closed = False

    @property
    def descriptor(self) -> ArenaDescriptor:
        return self._descriptor

    @property
    def name(self) -> str:
        return self._descriptor.name

    @property
    def num_slots(self) -> int:
        return self._descriptor.num_slots

    @property
    def slot_stride(self) -> int:
        return self._descriptor.slot_stride

    @property
    def total_bytes(self) -> int:
        return self._descriptor.total_bytes

    @property
    def spec(self) -> BufferSpec:
        return self._descriptor.spec

    def get_view(self, slot_id: int) -> np.ndarray:
        if self._closed or self._shm is None:
            raise PipelineClosedError("SharedMemoryArena is closed")
        return self._views[slot_id]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._views.clear()
        shm = self._shm
        self._shm = None
        if shm is not None:
            try:
                shm.close()
            except (BufferError, OSError) as exc:
                logger.warning("Error closing SharedMemoryArena %s: %s", self._descriptor.name, exc)
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def __enter__(self) -> "SharedMemoryArena":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
