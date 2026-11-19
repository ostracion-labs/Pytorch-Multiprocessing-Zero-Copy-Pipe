from collections import deque
from dataclasses import dataclass
import logging
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event as MpEvent
from queue import Empty
import signal
import traceback
from types import ModuleType
from typing import Callable, Deque, List, Optional

import torch

from zero_copy_pipe.dto import PacketDto
from zero_copy_pipe.interfaces import IGpuProgram
from zero_copy_pipe.pipeline.active_group import ActiveGroupTracker
from zero_copy_pipe.pipeline.shm_arena import ArenaDescriptor

logger = logging.getLogger(__name__)

CUDA_HOST_REGISTER_DEFAULT = 0


@dataclass(slots=True)
class _InFlightJob:
    pkt: PacketDto
    in_id: int
    out_id: int
    stream_slot: int
    h2d_event: Optional[torch.cuda.Event]
    done_event: Optional[torch.cuda.Event]
    in_released: bool = False


def _build_slot_tensors(shm: SharedMemory, desc: ArenaDescriptor) -> List[torch.Tensor]:
    """Create typed zero-copy slot views without leaking loop variables on `shm.buf`."""
    tdtype = desc.spec.torch_dtype()
    numel = desc.spec.numel
    shape = desc.spec.shape
    return [
        torch.frombuffer(
            shm.buf,
            dtype=tdtype,
            count=numel,
            offset=desc.slot_offset(idx),
        ).view(shape)
        for idx in range(desc.num_slots)
    ]


def _register_arena(
    cudart: ModuleType,
    shm: SharedMemory,
    desc: ArenaDescriptor,
    label: str,
) -> int:
    """Pin one contiguous SharedMemoryArena in this worker's CUDA virtual address space."""
    raw_view = torch.frombuffer(shm.buf, dtype=torch.uint8, count=desc.total_bytes)
    ptr = raw_view.data_ptr()
    del raw_view
    err = cudart.cudaHostRegister(ptr, desc.total_bytes, CUDA_HOST_REGISTER_DEFAULT)
    if int(err) != 0:
        raise RuntimeError(
            f"cudaHostRegister({label}) failed: {cudart.cudaGetErrorString(err)}"
        )
    return ptr


def gpu_worker_loop(
    worker_id: int,
    program_factory: Callable[[], IGpuProgram],
    in_arena_desc: ArenaDescriptor,
    out_arena_desc: ArenaDescriptor,
    in_work_q: "mp.Queue[Optional[PacketDto]]",
    in_free_q: "mp.Queue[int]",
    out_work_q: "mp.Queue[Optional[PacketDto]]",
    out_free_q: "mp.Queue[int]",
    active_group: ActiveGroupTracker,
    ready_event: MpEvent,
    abort_event: MpEvent,
    error_q: "mp.Queue[tuple[str, str]]",
    pin_memory: bool = True,
    pinned_status_q: Optional["mp.Queue[tuple[int, bool, bool]]"] = None,
) -> None:
    """GPU worker process: pins contiguous SHM arenas once and runs `IGpuProgram`."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    program: Optional[IGpuProgram] = None
    cudart: Optional[ModuleType] = None
    cuda_initialized = False
    in_shm: Optional[SharedMemory] = None
    out_shm: Optional[SharedMemory] = None
    in_tensors: List[torch.Tensor] = []
    out_tensors: List[torch.Tensor] = []
    pinned_ptrs: List[int] = []
    in_flight: Deque[_InFlightJob] = deque()
    pending_pkt: Optional[PacketDto] = None
    pending_out_id: Optional[int] = None

    try:
        program = program_factory()
        if program.input_spec != in_arena_desc.spec:
            raise ValueError(
                f"Program input_spec {program.input_spec} != arena spec {in_arena_desc.spec}"
            )
        if program.output_spec != out_arena_desc.spec:
            raise ValueError(
                f"Program output_spec {program.output_spec} != arena spec {out_arena_desc.spec}"
            )

        program.setup()
        program.warmup()

        should_pin = pin_memory and program.uses_cuda
        if should_pin:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA pinning requested but CUDA is not available")
            torch.cuda.init()
            _ = torch.empty(0, device="cuda")
            cuda_initialized = True
            cudart = torch.cuda.cudart()  # type: ignore[assignment]
        elif program.uses_cuda:
            cuda_initialized = True

        in_shm = SharedMemory(name=in_arena_desc.name, create=False, track=False)
        out_shm = SharedMemory(name=out_arena_desc.name, create=False, track=False)

        if cudart is not None:
            pinned_ptrs.append(_register_arena(cudart, in_shm, in_arena_desc, "input"))
            pinned_ptrs.append(_register_arena(cudart, out_shm, out_arena_desc, "output"))

        in_tensors = _build_slot_tensors(in_shm, in_arena_desc)
        out_tensors = _build_slot_tensors(out_shm, out_arena_desc)

        if cudart is not None:
            all_in_pinned = all(t.is_pinned() for t in in_tensors)
            all_out_pinned = all(t.is_pinned() for t in out_tensors)
            if not (all_in_pinned and all_out_pinned):
                raise RuntimeError(
                    f"SharedMemoryArena tensors not pinned as expected: "
                    f"in={all_in_pinned}, out={all_out_pinned}"
                )
            if pinned_status_q is not None:
                pinned_status_q.put((worker_id, all_in_pinned, all_out_pinned))
        elif pinned_status_q is not None:
            pinned_status_q.put((worker_id, False, False))

        ready_event.set()

        effective_streams = program.num_streams if program.uses_cuda else 1
        free_stream_slots: Deque[int] = deque(range(effective_streams))
        saw_sentinel = False

        def _finish_oldest() -> None:
            job = in_flight.popleft()
            if job.done_event is not None:
                job.done_event.synchronize()
            if not job.in_released:
                in_free_q.put(job.in_id)
                job.in_released = True
            free_stream_slots.append(job.stream_slot)
            if active_group.is_stale(job.pkt.group_id):
                out_free_q.put(job.out_id)
                active_group.record_skipped()
            else:
                out_work_q.put(
                    PacketDto(
                        group_id=job.pkt.group_id,
                        group_index=job.pkt.group_index,
                        dimension=out_arena_desc.spec.shape,
                        data_id=job.out_id,
                    )
                )

        def _poll_in_flight() -> None:
            for job in in_flight:
                if not job.in_released and (
                    job.h2d_event is None or job.h2d_event.query()
                ):
                    in_free_q.put(job.in_id)
                    job.in_released = True
            while in_flight and (
                in_flight[0].done_event is None or in_flight[0].done_event.query()
            ):
                _finish_oldest()

        while not abort_event.is_set():
            _poll_in_flight()

            if pending_pkt is None and not saw_sentinel:
                try:
                    item = (
                        in_work_q.get_nowait()
                        if in_flight
                        else in_work_q.get(timeout=0.05)
                    )
                except Empty:
                    if in_flight:
                        _finish_oldest()
                    continue

                if item is None:
                    saw_sentinel = True
                elif active_group.is_stale(item.group_id):
                    in_free_q.put(item.data_id)
                    active_group.record_skipped()
                    continue
                else:
                    pending_pkt = item

            if saw_sentinel:
                while in_flight and not abort_event.is_set():
                    _finish_oldest()
                break

            if not free_stream_slots:
                _finish_oldest()
                continue

            assert pending_pkt is not None
            if pending_out_id is None:
                try:
                    pending_out_id = (
                        out_free_q.get_nowait()
                        if in_flight
                        else out_free_q.get(timeout=0.05)
                    )
                except Empty:
                    if in_flight:
                        _finish_oldest()
                    continue

            if active_group.is_stale(pending_pkt.group_id):
                in_free_q.put(pending_pkt.data_id)
                out_free_q.put(pending_out_id)
                pending_pkt = None
                pending_out_id = None
                active_group.record_skipped()
                continue

            stream_slot = free_stream_slots.popleft()
            pkt = pending_pkt
            out_id = pending_out_id
            try:
                h2d_ev, done_ev = program.enqueue(
                    in_tensors[pkt.data_id],
                    out_tensors[out_id],
                    stream_slot=stream_slot,
                )
            except BaseException:
                free_stream_slots.append(stream_slot)
                raise

            pending_pkt = None
            pending_out_id = None
            in_flight.append(
                _InFlightJob(
                    pkt=pkt,
                    in_id=pkt.data_id,
                    out_id=out_id,
                    stream_slot=stream_slot,
                    h2d_event=h2d_ev,
                    done_event=done_ev,
                )
            )
            if done_ev is None:
                _finish_oldest()

    except BaseException as exc:
        if not (isinstance(exc, SystemExit) and exc.code in (None, 0)):
            try:
                error_q.put((f"gpu-worker-{worker_id}", traceback.format_exc()))
            except Exception:
                pass
        raise
    finally:
        if cuda_initialized:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        if pending_pkt is not None:
            try:
                in_free_q.put(pending_pkt.data_id)
            except Exception:
                pass
            pending_pkt = None
        if pending_out_id is not None:
            try:
                out_free_q.put(pending_out_id)
            except Exception:
                pass
            pending_out_id = None

        while in_flight:
            job = in_flight.popleft()
            if not job.in_released:
                try:
                    in_free_q.put(job.in_id)
                except Exception:
                    pass
            try:
                out_free_q.put(job.out_id)
            except Exception:
                pass

        if program is not None:
            try:
                program.close()
            except Exception as exc:
                logger.warning("program.close() failed in gpu-worker-%s: %s", worker_id, exc)

        if cudart is not None:
            for ptr in pinned_ptrs:
                try:
                    err = cudart.cudaHostUnregister(ptr)
                    if int(err) != 0:
                        logger.warning(
                            "cudaHostUnregister failed in gpu-worker-%s: %s",
                            worker_id,
                            cudart.cudaGetErrorString(err),
                        )
                except Exception as exc:
                    logger.warning(
                        "cudaHostUnregister error in gpu-worker-%s: %s",
                        worker_id,
                        exc,
                    )
            pinned_ptrs.clear()

        in_tensors.clear()
        out_tensors.clear()

        for shm in (in_shm, out_shm):
            if shm is not None:
                try:
                    shm.close()
                except Exception as exc:
                    logger.warning(
                        "Failed to close SharedMemory in gpu-worker-%s: %s",
                        worker_id,
                        exc,
                    )
