from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Any, Optional
import uuid

import numpy as np
import pytest
import torch

from zero_copy_pipe import (
    BufferSpec,
    DebayerConfig,
    DebayerProgramFactory,
    IGpuProgram,
    IOutputProcessor,
    PipeManager,
    PipelineWorkerError,
)


def _solid_rggb(height: int, width: int, r: int, g: int, b: int) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.uint8)
    img[0::2, 0::2] = r
    img[0::2, 1::2] = g
    img[1::2, 0::2] = g
    img[1::2, 1::2] = b
    return img


def _summarize_bgr(
    buf: np.ndarray, pkt
) -> tuple[int, int, tuple[int, ...], tuple[int, int, int]]:
    interior = buf[2:-2, 2:-2]
    mean_bgr = (
        int(round(float(interior[:, :, 0].mean()))),
        int(round(float(interior[:, :, 1].mean()))),
        int(round(float(interior[:, :, 2].mean()))),
    )
    return pkt.group_id, pkt.group_index, pkt.dimension, mean_bgr


def _attempt_mutate_view(buf: np.ndarray, pkt) -> int:
    buf[0, 0, 0] = 1
    return pkt.group_index


def _crashing_consumer(buf: np.ndarray, pkt) -> int:
    raise RuntimeError("intentional consumer crash")


class _CountingOutputProcessor(IOutputProcessor):
    def __init__(self, counter: Any, delay: float) -> None:
        self._counter = counter
        self._delay = delay

    def setup(self) -> None:
        pass

    def process(self, frame: np.ndarray, pkt) -> int:
        if self._delay > 0:
            time.sleep(self._delay)
        with self._counter.get_lock():
            self._counter.value += 1
        return pkt.group_index

    def close(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class _CountingOutputFactory:
    counter: Any
    delay: float = 0.05

    def __call__(self) -> IOutputProcessor:
        return _CountingOutputProcessor(self.counter, self.delay)


class _InvertGpuProgram(IGpuProgram):
    def __init__(self, height: int, width: int) -> None:
        self._in_spec = BufferSpec((height, width), "uint8")
        self._out_spec = BufferSpec((height, width), "uint8")
        self._ready = False

    @property
    def input_spec(self) -> BufferSpec:
        return self._in_spec

    @property
    def output_spec(self) -> BufferSpec:
        return self._out_spec

    @property
    def uses_cuda(self) -> bool:
        return False

    def setup(self) -> None:
        self._ready = True

    def warmup(self) -> None:
        assert self._ready

    def enqueue(self, host_input: torch.Tensor, host_output: torch.Tensor, stream_slot: int = 0):
        assert self._ready
        host_output.copy_(255 - host_input)
        return None, None

    def close(self) -> None:
        self._ready = False


@dataclass(frozen=True, slots=True)
class _InvertGpuFactory:
    height: int
    width: int

    @property
    def input_spec(self) -> BufferSpec:
        return BufferSpec((self.height, self.width), "uint8")

    @property
    def output_spec(self) -> BufferSpec:
        return BufferSpec((self.height, self.width), "uint8")

    def __call__(self) -> IGpuProgram:
        return _InvertGpuProgram(self.height, self.width)


class _StatefulScaleProcessor(IOutputProcessor):
    def __init__(self, factor: int) -> None:
        self._factor = factor
        self._initialized = False

    def setup(self) -> None:
        self._initialized = True

    def process(self, frame: np.ndarray, pkt) -> tuple[int, int]:
        assert self._initialized
        return pkt.group_index, int(frame.mean()) * self._factor

    def close(self) -> None:
        self._initialized = False


@dataclass(frozen=True, slots=True)
class _StatefulScaleFactory:
    factor: int

    def __call__(self) -> IOutputProcessor:
        return _StatefulScaleProcessor(self.factor)


def _broken_gpu_factory() -> IGpuProgram:
    raise RuntimeError("simulated gpu startup failure")


def _broken_cpu_factory() -> IOutputProcessor:
    raise RuntimeError("simulated cpu startup failure")


def _cpu_debayer_factory() -> DebayerProgramFactory:
    return DebayerProgramFactory(
        DebayerConfig(height=16, width=16, layout_name="RGGB", device="cpu", num_streams=1)
    )


def _cuda_debayer_factory() -> DebayerProgramFactory:
    return DebayerProgramFactory(
        DebayerConfig(height=16, width=16, layout_name="RGGB", device="cuda", num_streams=2)
    )


@pytest.mark.parametrize(
    ("factory_fn", "pin_memory"),
    [
        (_cpu_debayer_factory, False),
        pytest.param(
            _cuda_debayer_factory,
            True,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
def test_pipe_zero_copy_acquire_non_green_bgr_and_pinning(
    factory_fn, pin_memory: bool
) -> None:
    samples = [
        (200, 50, 30),
        (40, 180, 90),
        (10, 60, 220),
        (240, 120, 15),
        (80, 200, 140),
        (170, 30, 110),
    ]
    with PipeManager(
        gpu_program_factory=factory_fn(),
        output_processor=_summarize_bgr,
        num_input_slots=2,
        num_output_slots=2,
        num_cpu_workers=2,
        pin_memory=pin_memory,
    ) as pipe:
        if pin_memory:
            assert pipe.gpu_pinned_status == {0: (True, True)}
        else:
            assert pipe.gpu_pinned_status == {0: (False, False)}

        # Verify slot rollback on cancel() and exception inside acquire_input
        with pipe.acquire_input(group_id=1, group_index=999) as slot:
            slot.buffer[:] = 0
            slot.cancel()

        with pytest.raises(RuntimeError, match="abort fill"):
            with pipe.acquire_input(group_id=1, group_index=998) as slot:
                slot.buffer[:] = 0
                raise RuntimeError("abort fill")

        # Now run all 6 frames through the 2-slot pool; proves cancelled/failed slots returned cleanly
        for idx, (r, g, b) in enumerate(samples):
            with pipe.acquire_input(group_id=42, group_index=idx) as slot:
                np.copyto(slot.buffer, _solid_rggb(16, 16, r=r, g=g, b=b))

        results = sorted(
            (pipe.get_result(timeout=15.0) for _ in range(len(samples))),
            key=lambda item: item[1],
        )

    assert len(results) == len(samples)
    for idx, (r, g, b) in enumerate(samples):
        gid, gidx, dim, mean_bgr = results[idx]
        assert gid == 42
        assert gidx == idx
        assert dim == (16, 16, 3)
        assert mean_bgr == (b, g, r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pipe_no_deadlock_single_output_slot_with_two_streams() -> None:
    num_frames = 5
    with PipeManager(
        gpu_program_factory=_cuda_debayer_factory(),
        output_processor=_summarize_bgr,
        num_input_slots=2,
        num_output_slots=1,
        num_cpu_workers=1,
        pin_memory=True,
    ) as pipe:
        for idx in range(num_frames):
            pipe.submit(_solid_rggb(16, 16, r=150, g=80, b=30), group_id=7, group_index=idx)

        results = [pipe.get_result(timeout=15.0) for _ in range(num_frames)]

    assert len(results) == num_frames
    assert all(res[3] == (30, 80, 150) for res in results)


def test_pipe_swappable_custom_gpu_and_stateful_cpu_ends() -> None:
    factory = _InvertGpuFactory(height=8, width=10)
    with PipeManager(
        gpu_program_factory=factory,
        output_processor_factory=_StatefulScaleFactory(factor=2),
        num_input_slots=2,
        num_output_slots=2,
        num_cpu_workers=2,
        pin_memory=False,
    ) as pipe:
        for idx, val in enumerate([10, 55, 200]):
            with pipe.acquire_input(group_id=9, group_index=idx) as slot:
                slot.buffer.fill(val)

        results = sorted((pipe.get_result(timeout=15.0) for _ in range(3)), key=lambda x: x[0])

    assert results == [(0, (255 - 10) * 2), (1, (255 - 55) * 2), (2, (255 - 200) * 2)]


def test_pipe_stale_group_cancellation_128bit() -> None:
    gid_stale = uuid.uuid4().int
    gid_active = uuid.uuid4().int
    gid_resumed = uuid.uuid4().int

    with PipeManager(
        gpu_program_factory=_cpu_debayer_factory(),
        output_processor=_summarize_bgr,
        num_input_slots=2,
        num_output_slots=2,
        num_cpu_workers=1,
        pin_memory=False,
    ) as pipe:
        pipe.set_active_group(gid_active)
        assert pipe.active_group_id == gid_active

        # Submit 3 stale frames followed by 2 active frames
        for idx in range(3):
            pipe.submit(_solid_rggb(16, 16, 10, 20, 30), group_id=gid_stale, group_index=idx)
        for idx in range(2):
            pipe.submit(_solid_rggb(16, 16, 90, 40, 120), group_id=gid_active, group_index=idx)

        active_results = [pipe.get_result(timeout=15.0) for _ in range(2)]
        assert [r[0] for r in active_results] == [gid_active, gid_active]
        assert sorted(r[1] for r in active_results) == [0, 1]
        assert pipe.skipped_count == 3

        # Clear active group and verify subsequent different group_ids process normally
        pipe.clear_active_group()
        assert pipe.active_group_id is None
        pipe.submit(_solid_rggb(16, 16, 15, 25, 35), group_id=gid_resumed, group_index=0)
        resumed = pipe.get_result(timeout=15.0)
        assert resumed[0] == gid_resumed
        assert resumed[3] == (35, 25, 15)


def test_pipe_readonly_consumer_view_raises_worker_error() -> None:
    with pytest.raises(PipelineWorkerError) as exc_info:
        with PipeManager(
            gpu_program_factory=_cpu_debayer_factory(),
            output_processor=_attempt_mutate_view,
            num_input_slots=2,
            num_output_slots=2,
            num_cpu_workers=1,
            pin_memory=False,
        ) as pipe:
            pipe.submit(_solid_rggb(16, 16, 10, 20, 30), group_id=1, group_index=0)
            pipe.get_result(timeout=15.0)

    assert "read-only" in exc_info.value.remote_traceback.lower()


def test_pipe_startup_crash_fails_fast() -> None:
    # GPU worker imports torch during spawn (~5-7s on WSL2), so verify well below startup_timeout=60s
    t0 = time.monotonic()
    with pytest.raises(PipelineWorkerError) as exc_info:
        pipe = PipeManager(
            gpu_program_factory=_broken_gpu_factory,
            output_processor=_summarize_bgr,
            input_spec=BufferSpec((16, 16), "uint8"),
            output_spec=BufferSpec((16, 16, 3), "uint8"),
            num_input_slots=2,
            num_output_slots=2,
            num_cpu_workers=1,
            pin_memory=False,
            startup_timeout=60.0,
        )
        pipe.start()

    elapsed_gpu = time.monotonic() - t0
    assert elapsed_gpu < 20.0
    assert "simulated gpu startup failure" in exc_info.value.remote_traceback

    # CPU-only worker failure with torch-free factory fails almost immediately (< 5s)
    t1 = time.monotonic()
    with pytest.raises(PipelineWorkerError) as cpu_exc:
        pipe2 = PipeManager(
            gpu_program_factory=_cpu_debayer_factory(),
            output_processor_factory=_broken_cpu_factory,
            num_input_slots=2,
            num_output_slots=2,
            num_cpu_workers=1,
            pin_memory=False,
            startup_timeout=60.0,
        )
        pipe2.start()

    elapsed_cpu = time.monotonic() - t1
    assert elapsed_cpu < 15.0
    assert "simulated cpu startup failure" in cpu_exc.value.remote_traceback


def test_pipe_midrun_crash_propagates_without_hang() -> None:
    with pytest.raises(PipelineWorkerError) as exc_info:
        with PipeManager(
            gpu_program_factory=_cpu_debayer_factory(),
            output_processor=_crashing_consumer,
            num_input_slots=2,
            num_output_slots=2,
            num_cpu_workers=1,
            pin_memory=False,
        ) as pipe:
            pipe.submit(_solid_rggb(16, 16, 10, 20, 30), group_id=1, group_index=0)
            pipe.get_result(timeout=10.0)

    assert "intentional consumer crash" in exc_info.value.remote_traceback


def test_pipe_graceful_drain_and_forced_abort() -> None:
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)

    pipe = PipeManager(
        gpu_program_factory=_cpu_debayer_factory(),
        output_processor_factory=_CountingOutputFactory(counter=counter, delay=0.05),
        num_input_slots=4,
        num_output_slots=4,
        num_cpu_workers=2,
        pin_memory=False,
        collect_results=False,
    )
    pipe.start()
    for idx in range(5):
        pipe.submit(_solid_rggb(16, 16, 50, 60, 70), group_id=1, group_index=idx)
    pipe.shutdown(drain=True, timeout=10.0)
    assert counter.value == 5

    # Now test forced abort under queued backlog without deadlock
    counter2 = ctx.Value("i", 0)
    pipe2 = PipeManager(
        gpu_program_factory=_cpu_debayer_factory(),
        output_processor_factory=_CountingOutputFactory(counter=counter2, delay=0.3),
        num_input_slots=2,
        num_output_slots=2,
        num_cpu_workers=1,
        pin_memory=False,
        collect_results=False,
    )
    pipe2.start()
    for idx in range(4):
        pipe2.submit(_solid_rggb(16, 16, 50, 60, 70), group_id=2, group_index=idx)
    t0 = time.monotonic()
    pipe2.shutdown(drain=False, timeout=5.0)
    assert time.monotonic() - t0 < 4.0
