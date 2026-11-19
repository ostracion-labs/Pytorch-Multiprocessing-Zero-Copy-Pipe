from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as MpEvent
from queue import Empty
import time
from types import TracebackType
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Type

import numpy as np

from zero_copy_pipe.dto import BufferSpec, PacketDto
from zero_copy_pipe.interfaces import (
    CallableOutputProcessorFactory,
    IGpuProgram,
    IOutputProcessor,
)
from zero_copy_pipe.pipeline.active_group import MAX_GROUP_ID, ActiveGroupTracker
from zero_copy_pipe.pipeline.cpu_worker import cpu_worker_loop
from zero_copy_pipe.pipeline.errors import (
    PipelineClosedError,
    PipelineError,
    PipelineWorkerError,
)
from zero_copy_pipe.pipeline.shm_arena import SharedMemoryArena


@dataclass(slots=True)
class InputSlot:
    """Zero-copy writable input slot leased from `PipeManager.acquire_input`."""

    slot_id: int
    buffer: np.ndarray
    group_id: int
    group_index: int
    _cancelled: bool = False

    def cancel(self) -> None:
        """Abort submission of this slot and return it to the free pool on exit."""
        self._cancelled = True


class PipeManager(AbstractContextManager):
    """Orchestrates a 3-end zero-copy multiprocess pipe:

    End 1 (Input producer) -> End 2 (GPU worker(s)) -> End 3 (Downstream CPU workers).
    """

    def __init__(
        self,
        gpu_program_factory: Callable[[], IGpuProgram],
        output_processor: Optional[Callable[[np.ndarray, PacketDto], Any]] = None,
        *,
        output_processor_factory: Optional[Callable[[], IOutputProcessor]] = None,
        input_spec: Optional[BufferSpec] = None,
        output_spec: Optional[BufferSpec] = None,
        num_input_slots: int = 4,
        num_output_slots: int = 4,
        num_gpu_workers: int = 1,
        num_cpu_workers: int = 2,
        pin_memory: bool = True,
        collect_results: bool = True,
        startup_timeout: float = 30.0,
    ) -> None:
        if (output_processor is None) == (output_processor_factory is None):
            raise ValueError(
                "Provide exactly one of `output_processor` or `output_processor_factory`"
            )
        if num_input_slots <= 0:
            raise ValueError(f"num_input_slots must be positive, got {num_input_slots}")
        if num_output_slots <= 0:
            raise ValueError(f"num_output_slots must be positive, got {num_output_slots}")
        if num_gpu_workers <= 0:
            raise ValueError(f"num_gpu_workers must be positive, got {num_gpu_workers}")
        if num_cpu_workers <= 0:
            raise ValueError(f"num_cpu_workers must be positive, got {num_cpu_workers}")
        if startup_timeout <= 0:
            raise ValueError(f"startup_timeout must be positive, got {startup_timeout}")

        resolved_in = (
            input_spec
            if input_spec is not None
            else getattr(gpu_program_factory, "input_spec", None)
        )
        resolved_out = (
            output_spec
            if output_spec is not None
            else getattr(gpu_program_factory, "output_spec", None)
        )
        if not isinstance(resolved_in, BufferSpec):
            raise ValueError(
                "input_spec could not be resolved; pass `input_spec=BufferSpec(...)` "
                "or expose `.input_spec` on `gpu_program_factory`"
            )
        if not isinstance(resolved_out, BufferSpec):
            raise ValueError(
                "output_spec could not be resolved; pass `output_spec=BufferSpec(...)` "
                "or expose `.output_spec` on `gpu_program_factory`"
            )

        self._gpu_program_factory = gpu_program_factory
        self._processor_factory: Callable[[], IOutputProcessor] = (
            output_processor_factory
            if output_processor_factory is not None
            else CallableOutputProcessorFactory(output_processor)  # type: ignore[arg-type]
        )
        self._input_spec = resolved_in
        self._output_spec = resolved_out
        self._num_gpu_workers = num_gpu_workers
        self._num_cpu_workers = num_cpu_workers
        self._pin_memory = pin_memory
        self._collect_results = collect_results
        self._startup_timeout = startup_timeout

        self._ctx: SpawnContext = mp.get_context("spawn")
        self._in_arena = SharedMemoryArena(num_input_slots, self._input_spec)
        try:
            self._out_arena = SharedMemoryArena(num_output_slots, self._output_spec)
        except BaseException:
            self._in_arena.close()
            raise

        self._in_work_q: mp.Queue[Optional[PacketDto]] = self._ctx.Queue()
        self._in_free_q: mp.Queue[int] = self._ctx.Queue()
        self._out_work_q: mp.Queue[Optional[PacketDto]] = self._ctx.Queue()
        self._out_free_q: mp.Queue[int] = self._ctx.Queue()
        self._result_q: Optional[mp.Queue[Any]] = (
            self._ctx.Queue() if collect_results else None
        )
        self._error_q: mp.Queue[tuple[str, str]] = self._ctx.Queue()
        self._pinned_status_q: mp.Queue[tuple[int, bool, bool]] = self._ctx.Queue()

        self._queues: List[mp.Queue[Any]] = [
            self._in_work_q,
            self._in_free_q,
            self._out_work_q,
            self._out_free_q,
            self._error_q,
            self._pinned_status_q,
        ]
        if self._result_q is not None:
            self._queues.append(self._result_q)

        for slot_id in range(num_input_slots):
            self._in_free_q.put(slot_id)
        for slot_id in range(num_output_slots):
            self._out_free_q.put(slot_id)

        self._active_group = ActiveGroupTracker(self._ctx)
        self._abort_event: MpEvent = self._ctx.Event()

        self._gpu_procs: List[BaseProcess] = []
        self._cpu_procs: List[BaseProcess] = []
        self._gpu_pinned_status: Dict[int, Tuple[bool, bool]] = {}
        self._errors: List[Tuple[str, str]] = []
        self._started = False
        self._closed = False

    @property
    def input_spec(self) -> BufferSpec:
        return self._input_spec

    @property
    def output_spec(self) -> BufferSpec:
        return self._output_spec

    @property
    def gpu_pinned_status(self) -> Dict[int, Tuple[bool, bool]]:
        return dict(self._gpu_pinned_status)

    @property
    def active_group_id(self) -> Optional[int]:
        return self._active_group.active_group_id

    @property
    def skipped_count(self) -> int:
        return self._active_group.skipped_count

    @property
    def errors(self) -> List[Tuple[str, str]]:
        self._drain_error_queue()
        return list(self._errors)

    def set_active_group(self, group_id: int) -> None:
        self._active_group.set_active_group(group_id)

    def clear_active_group(self) -> None:
        self._active_group.clear_active_group()

    def start(self) -> None:
        if self._closed:
            raise PipelineClosedError("PipeManager has already been closed")
        if self._started:
            return

        # Lazy import so importing PipeManager or spawning CPU workers never imports torch.
        from zero_copy_pipe.pipeline.gpu_worker import gpu_worker_loop

        gpu_ready_events: List[MpEvent] = [
            self._ctx.Event() for _ in range(self._num_gpu_workers)
        ]
        cpu_ready_events: List[MpEvent] = [
            self._ctx.Event() for _ in range(self._num_cpu_workers)
        ]

        try:
            for wid in range(self._num_gpu_workers):
                proc = self._ctx.Process(
                    target=gpu_worker_loop,
                    args=(
                        wid,
                        self._gpu_program_factory,
                        self._in_arena.descriptor,
                        self._out_arena.descriptor,
                        self._in_work_q,
                        self._in_free_q,
                        self._out_work_q,
                        self._out_free_q,
                        self._active_group,
                        gpu_ready_events[wid],
                        self._abort_event,
                        self._error_q,
                        self._pin_memory,
                        self._pinned_status_q,
                    ),
                )
                self._gpu_procs.append(proc)
                proc.start()

            for wid in range(self._num_cpu_workers):
                proc = self._ctx.Process(
                    target=cpu_worker_loop,
                    args=(
                        wid,
                        self._out_arena.descriptor,
                        self._out_work_q,
                        self._out_free_q,
                        self._active_group,
                        cpu_ready_events[wid],
                        self._abort_event,
                        self._error_q,
                        self._processor_factory,
                        self._result_q,
                    ),
                )
                self._cpu_procs.append(proc)
                proc.start()

            deadline = time.monotonic() + self._startup_timeout
            while True:
                self._drain_error_queue()
                if self._errors:
                    err = self._build_worker_error()
                    self.shutdown(drain=False)
                    raise err

                self._check_premature_exit(self._gpu_procs, gpu_ready_events, "gpu-worker")
                self._check_premature_exit(self._cpu_procs, cpu_ready_events, "cpu-worker")

                if all(ev.is_set() for ev in gpu_ready_events) and all(
                    ev.is_set() for ev in cpu_ready_events
                ):
                    break

                if time.monotonic() >= deadline:
                    self.shutdown(drain=False)
                    raise TimeoutError(
                        f"Workers failed to initialize within {self._startup_timeout}s"
                    )
                time.sleep(0.05)

            self._gpu_pinned_status.clear()
            for _ in range(self._num_gpu_workers):
                try:
                    wid, in_pinned, out_pinned = self._pinned_status_q.get(timeout=2.0)
                    self._gpu_pinned_status[wid] = (in_pinned, out_pinned)
                except Empty:
                    break

            self._started = True
        except BaseException:
            if not self._closed:
                self.shutdown(drain=False)
            raise

    def _check_premature_exit(
        self,
        procs: Sequence[BaseProcess],
        ready_events: Sequence[MpEvent],
        prefix: str,
    ) -> None:
        for idx, (proc, ev) in enumerate(zip(procs, ready_events)):
            if not ev.is_set() and not proc.is_alive():
                time.sleep(0.05)
                self._drain_error_queue()
                if not self._errors:
                    self._errors.append(
                        (
                            f"{prefix}-{idx}",
                            f"Exited prematurely during startup with exitcode={proc.exitcode}",
                        )
                    )
                raise self._build_worker_error()

    def _drain_error_queue(self) -> None:
        if self._closed:
            return
        while True:
            try:
                self._errors.append(self._error_q.get_nowait())
            except (Empty, ValueError, OSError):
                break

    def _build_worker_error(self) -> PipelineWorkerError:
        if not self._errors:
            return PipelineWorkerError("unknown-worker", "Worker exited abnormally")
        if len(self._errors) == 1:
            name, tb = self._errors[0]
            return PipelineWorkerError(name, tb)
        combined = "\n---\n".join(f"[{name}]\n{tb}" for name, tb in self._errors)
        return PipelineWorkerError(f"{len(self._errors)} workers", combined)

    def check_health(self) -> None:
        self._drain_error_queue()
        if self._errors:
            raise self._build_worker_error()
        if not self._started:
            return
        for prefix, procs in (("gpu-worker", self._gpu_procs), ("cpu-worker", self._cpu_procs)):
            for idx, proc in enumerate(procs):
                if not proc.is_alive():
                    time.sleep(0.05)
                    self._drain_error_queue()
                    if not self._errors:
                        self._errors.append(
                            (
                                f"{prefix}-{idx}",
                                f"Exited unexpectedly with exitcode={proc.exitcode}",
                            )
                        )
                    raise self._build_worker_error()

    def _pop_free_input_slot(self, timeout: Optional[float]) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self.check_health()
            try:
                return self._in_free_q.get(timeout=0.05)
            except Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for a free input SharedMemory slot")

    @contextmanager
    def acquire_input(
        self,
        group_id: int,
        group_index: int,
        timeout: Optional[float] = None,
    ) -> Iterator[InputSlot]:
        """Lease a writable zero-copy input slot and submit it to the pipe on exit."""
        if self._closed:
            raise PipelineClosedError("PipeManager is closed")
        if not self._started:
            raise PipelineError("PipeManager is not started")
        if (
            not isinstance(group_id, int)
            or isinstance(group_id, bool)
            or not (0 <= group_id <= MAX_GROUP_ID)
        ):
            raise ValueError(f"group_id must be in [0, 2^128 - 1], got {group_id}")
        self.check_health()

        slot_id = self._pop_free_input_slot(timeout=timeout)
        view = self._in_arena.get_view(slot_id)
        slot = InputSlot(
            slot_id=slot_id,
            buffer=view,
            group_id=group_id,
            group_index=group_index,
        )
        committed = False
        try:
            yield slot
            if not slot._cancelled:
                self.check_health()
                self._in_work_q.put(
                    PacketDto(
                        group_id=group_id,
                        group_index=group_index,
                        dimension=self._input_spec.shape,
                        data_id=slot_id,
                    )
                )
                committed = True
        finally:
            if not committed:
                try:
                    self._in_free_q.put(slot_id)
                except Exception:
                    pass

    def submit(
        self,
        frame: np.ndarray,
        group_id: int,
        group_index: int,
        timeout: Optional[float] = None,
    ) -> None:
        """Convenience wrapper copying `frame` into a leased input slot."""
        if frame.shape != self._input_spec.shape:
            raise ValueError(
                f"Expected frame shape {self._input_spec.shape}, got {frame.shape}"
            )
        if frame.dtype != self._input_spec.numpy_dtype:
            raise ValueError(
                f"Expected frame dtype {self._input_spec.numpy_dtype}, got {frame.dtype}"
            )
        with self.acquire_input(group_id, group_index, timeout=timeout) as slot:
            np.copyto(slot.buffer, frame)

    def get_result(self, timeout: Optional[float] = None) -> Any:
        """Retrieve one result emitted by downstream CPU workers."""
        if self._result_q is None:
            raise PipelineError("collect_results=False; results are not collected")
        if self._closed:
            raise PipelineClosedError("PipeManager is closed")

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self.check_health()
            try:
                return self._result_q.get(timeout=0.05)
            except Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for downstream result")

    def _join_procs(self, procs: Sequence[BaseProcess], timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        for proc in procs:
            remaining = max(0.0, deadline - time.monotonic())
            if proc.is_alive():
                proc.join(timeout=remaining)
        return all(not proc.is_alive() for proc in procs)

    def _terminate_stragglers(
        self, procs: Sequence[BaseProcess], prefix: str
    ) -> set[str]:
        terminated: set[str] = set()
        for idx, proc in enumerate(procs):
            if proc.is_alive():
                name = f"{prefix}-{idx}"
                terminated.add(name)
                proc.terminate()
                proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)
        return terminated

    def shutdown(self, drain: bool = True, timeout: float = 10.0) -> None:
        if self._closed:
            return

        self._started = False
        terminated_names: set[str] = set()
        try:
            if drain:
                for proc in self._gpu_procs:
                    if proc.is_alive():
                        try:
                            self._in_work_q.put(None)
                        except Exception:
                            pass
                if not self._join_procs(self._gpu_procs, timeout):
                    self._abort_event.set()
                    self._join_procs(self._gpu_procs, 2.0)

                for proc in self._cpu_procs:
                    if proc.is_alive():
                        try:
                            self._out_work_q.put(None)
                        except Exception:
                            pass
                if not self._join_procs(self._cpu_procs, timeout):
                    self._abort_event.set()
                    self._join_procs(self._cpu_procs, 2.0)
            else:
                self._abort_event.set()
                self._join_procs(self._gpu_procs + self._cpu_procs, timeout)

            terminated_names.update(
                self._terminate_stragglers(self._gpu_procs, "gpu-worker")
            )
            terminated_names.update(
                self._terminate_stragglers(self._cpu_procs, "cpu-worker")
            )

            self._drain_error_queue()
            recorded_names = {name for name, _ in self._errors}
            for prefix, procs in (
                ("gpu-worker", self._gpu_procs),
                ("cpu-worker", self._cpu_procs),
            ):
                for idx, proc in enumerate(procs):
                    name = f"{prefix}-{idx}"
                    if (
                        name not in terminated_names
                        and name not in recorded_names
                        and proc.exitcode not in (0, None)
                    ):
                        self._errors.append(
                            (name, f"Exited abnormally with exitcode={proc.exitcode}")
                        )
        finally:
            self._closed = True
            self._gpu_procs.clear()
            self._cpu_procs.clear()
            for queue in self._queues:
                try:
                    queue.close()
                except Exception:
                    pass
                try:
                    queue.cancel_join_thread()
                except Exception:
                    pass
            self._in_arena.close()
            self._out_arena.close()

    def __enter__(self) -> "PipeManager":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.shutdown(drain=(exc_type is None))
        if exc_type is None and self._errors:
            raise self._build_worker_error()
