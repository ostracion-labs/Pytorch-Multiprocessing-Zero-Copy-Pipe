from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from zero_copy_pipe import BufferSpec, PipelineClosedError, SharedMemoryArena
from zero_copy_pipe.pipeline.shm_arena import PAGE_SIZE


def test_shm_arena_page_alignment_zero_copy_and_unlink() -> None:
    spec = BufferSpec((8, 12), "uint8")
    with SharedMemoryArena(num_slots=3, spec=spec) as arena:
        desc = arena.descriptor
        assert arena.slot_stride == PAGE_SIZE
        assert arena.total_bytes == 3 * PAGE_SIZE
        assert desc.slot_offset(0) == 0
        assert desc.slot_offset(1) == PAGE_SIZE
        assert desc.slot_offset(2) == 2 * PAGE_SIZE

        with pytest.raises(IndexError):
            desc.slot_offset(3)

        slot1 = arena.get_view(1)
        slot1[:] = 77
        assert np.all(arena.get_view(0) == 0)
        assert np.all(arena.get_view(2) == 0)

        # Verify zero-copy visibility from another SharedMemory handle at slot 1's page offset
        other = SharedMemory(name=desc.name, create=False, track=False)
        try:
            other_view = np.ndarray(
                spec.shape,
                dtype=spec.numpy_dtype,
                buffer=other.buf,
                offset=desc.slot_offset(1),
            )
            assert np.all(other_view == 77)
            del other_view
        finally:
            other.close()

    # Idempotent close + PipelineClosedError on access
    arena.close()
    with pytest.raises(PipelineClosedError):
        arena.get_view(0)

    # Underlying POSIX segment must be unlinked
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=desc.name, create=False)
