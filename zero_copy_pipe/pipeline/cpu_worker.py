from contextlib import AbstractContextManager
import logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event as MpEvent
from queue import Empty
import signal
import traceback
from types import TracebackType
from typing import Any, Callable, List, Optional, Type

import numpy as np

from zero_copy_pipe.dto import PacketDto
from zero_copy_pipe.interfaces import IOutputProcessor
from zero_copy_pipe.pipeline.active_group import ActiveGroupTracker
from zero_copy_pipe.pipeline.errors import PipelineClosedError
from zero_copy_pipe.pipeline.shm_arena import ArenaDescriptor

logger = logging.getLogger(__name__)


class DownstreamArenaView(AbstractContextManager):
    """Attaches a downstream CPU worker to an output `SharedMemoryArena` with read-only views."""

    def __init__(self, arena_desc: ArenaDescriptor) -> None:
        shm = SharedMemory(name=arena_desc.name, create=False, track=False)
        views: List[np.ndarray] = []
        try:
            np_dtype = arena_desc.spec.numpy_dtype
            shape = arena_desc.spec.shape
            for idx in range(arena_desc.num_slots):
                view = np.ndarray(
                    shape,
                    dtype=np_dtype,
                    buffer=shm.buf,
                    offset=arena_desc.slot_offset(idx),
                )
                view.flags.writeable = False
                views.append(view)
        except BaseException:
            views.clear()
            try:
                shm.close()
            except Exception:
                pass
            raise

        self._shm: Optional[SharedMemory] = shm
        self._views: List[np.ndarray] = views
        self._closed = False

    def get_view(self, slot_id: int) -> np.ndarray:
        if self._closed or self._shm is None:
            raise PipelineClosedError("DownstreamArenaView is closed")
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
                logger.warning("Error closing DownstreamArenaView SharedMemory: %s", exc)

    def __enter__(self) -> "DownstreamArenaView":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()


def cpu_worker_loop(
    worker_id: int,
    out_arena_desc: ArenaDescriptor,
    work_q: "mp.Queue[Optional[PacketDto]]",
    free_q: "mp.Queue[int]",
    active_group: ActiveGroupTracker,
    ready_event: MpEvent,
    abort_event: MpEvent,
    error_q: "mp.Queue[tuple[str, str]]",
    processor_factory: Callable[[], IOutputProcessor],
    result_q: Optional["mp.Queue[Any]"] = None,
) -> None:
    """Downstream CPU worker: processes read-only output SHM frames and returns lease tokens."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        with DownstreamArenaView(out_arena_desc) as view_mgr, processor_factory() as processor:
            ready_event.set()

            while not abort_event.is_set():
                try:
                    pkt = work_q.get(timeout=0.05)
                except Empty:
                    continue

                if pkt is None:
                    break

                try:
                    if active_group.is_stale(pkt.group_id):
                        active_group.record_skipped()
                        continue
                    buf = view_mgr.get_view(pkt.data_id)
                    result = processor.process(buf, pkt)
                    if result_q is not None:
                        result_q.put(result)
                finally:
                    free_q.put(pkt.data_id)

    except BaseException as exc:
        if not (isinstance(exc, SystemExit) and exc.code in (None, 0)):
            try:
                error_q.put((f"cpu-worker-{worker_id}", traceback.format_exc()))
            except Exception:
                pass
        raise
