import ctypes
import multiprocessing as mp
from multiprocessing.context import BaseContext
from typing import Any, Optional

MAX_GROUP_ID = (1 << 128) - 1
_MASK_64 = (1 << 64) - 1


class ActiveGroupTracker:
    """Cross-process 128-bit active group tracker for real-time stale-frame cancellation."""

    def __init__(
        self,
        ctx: Optional[BaseContext] = None,
        _state: Optional[Any] = None,
        _skipped: Optional[Any] = None,
    ) -> None:
        if _state is not None and _skipped is not None:
            self._state = _state
            self._skipped = _skipped
            return

        mp_ctx = ctx if ctx is not None else mp.get_context("spawn")
        # Layout: [enabled (0 or 1), high_64_bits, low_64_bits]
        self._state = mp_ctx.Array(ctypes.c_uint64, 3, lock=True)
        self._skipped = mp_ctx.Value(ctypes.c_uint64, 0, lock=True)

    @classmethod
    def from_shared(cls, state: Any, skipped: Any) -> "ActiveGroupTracker":
        return cls(_state=state, _skipped=skipped)

    def set_active_group(self, group_id: int) -> None:
        if not isinstance(group_id, int) or isinstance(group_id, bool):
            raise ValueError(f"group_id must be an int, got {type(group_id).__name__}")
        if not (0 <= group_id <= MAX_GROUP_ID):
            raise ValueError(f"group_id must be in [0, 2^128 - 1], got {group_id}")

        hi = (group_id >> 64) & _MASK_64
        lo = group_id & _MASK_64
        with self._state.get_lock():
            self._state[0] = 1
            self._state[1] = hi
            self._state[2] = lo

    def clear_active_group(self) -> None:
        with self._state.get_lock():
            self._state[0] = 0
            self._state[1] = 0
            self._state[2] = 0

    @property
    def active_group_id(self) -> Optional[int]:
        with self._state.get_lock():
            if self._state[0] == 0:
                return None
            return (int(self._state[1]) << 64) | int(self._state[2])

    def is_stale(self, group_id: int) -> bool:
        with self._state.get_lock():
            if self._state[0] == 0:
                return False
            active = (int(self._state[1]) << 64) | int(self._state[2])
            return active != group_id

    def record_skipped(self, count: int = 1) -> None:
        if count <= 0:
            return
        with self._skipped.get_lock():
            self._skipped.value += count

    @property
    def skipped_count(self) -> int:
        with self._skipped.get_lock():
            return int(self._skipped.value)
