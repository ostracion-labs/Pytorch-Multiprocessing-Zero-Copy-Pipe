from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class BufferSpec:
    """Pure-Python specification for a fixed-shape, fixed-dtype shared buffer slot."""

    shape: tuple[int, ...]
    dtype_str: str

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("BufferSpec.shape must be non-empty")
        normalized_shape = tuple(int(dim) for dim in self.shape)
        if any(dim <= 0 for dim in normalized_shape):
            raise ValueError(f"All dimensions in BufferSpec.shape must be positive, got {self.shape}")
        object.__setattr__(self, "shape", normalized_shape)

        try:
            np_dtype = np.dtype(self.dtype_str)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dtype: {self.dtype_str!r}") from exc
        if np_dtype.itemsize <= 0 or np_dtype.hasobject:
            raise ValueError(f"Unsupported non-numeric or zero-size dtype: {self.dtype_str!r}")
        object.__setattr__(self, "dtype_str", np_dtype.name)

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype_str)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * self.numpy_dtype.itemsize

    def torch_dtype(self) -> "torch.dtype":
        """Lazily resolve the matching PyTorch dtype without importing torch at module load."""
        import torch

        mapping = {
            "uint8": torch.uint8,
            "uint16": torch.uint16,
            "int16": torch.int16,
            "int32": torch.int32,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        try:
            return mapping[self.dtype_str]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported dtype {self.dtype_str!r} for torch tensor conversion; "
                f"supported: {sorted(mapping)}"
            ) from exc
