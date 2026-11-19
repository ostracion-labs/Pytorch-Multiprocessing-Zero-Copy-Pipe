import subprocess
import sys

import pytest

from zero_copy_pipe import DebayerConfig, DebayerProgramFactory, PipeManager


def _check_no_torch_in_worker(frame, pkt):
    return "torch" in sys.modules


def test_core_imports_do_not_load_torch() -> None:
    code = (
        "import sys; "
        "import zero_copy_pipe; "
        "from zero_copy_pipe import ("
        "    ActiveGroupTracker, BufferSpec, DebayerConfig, DebayerProgramFactory, "
        "    IGpuProgram, IOutputProcessor, InputSlot, PacketDto, PipeManager, SharedMemoryArena"
        "); "
        "from zero_copy_pipe.pipeline import cpu_worker_loop, DownstreamArenaView; "
        "assert 'torch' not in sys.modules, f'torch loaded: {list(sys.modules)}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_cpu_worker_process_never_imports_torch() -> None:
    cfg = DebayerConfig(
        height=8,
        width=8,
        layout_name="RGGB",
        device="cpu",
        num_streams=1,
    )
    import numpy as np

    with PipeManager(
        gpu_program_factory=DebayerProgramFactory(cfg),
        output_processor=_check_no_torch_in_worker,
        num_input_slots=2,
        num_output_slots=2,
        num_cpu_workers=1,
        pin_memory=False,
    ) as pipe:
        pipe.submit(np.zeros((8, 8), dtype=np.uint8), group_id=1, group_index=0)
        torch_loaded = pipe.get_result(timeout=15.0)
        assert torch_loaded is False
