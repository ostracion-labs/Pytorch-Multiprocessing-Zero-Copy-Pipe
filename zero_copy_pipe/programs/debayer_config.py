from dataclasses import dataclass
from typing import TYPE_CHECKING

from zero_copy_pipe.dto import BufferSpec

if TYPE_CHECKING:
    from zero_copy_pipe.interfaces import IGpuProgram

VALID_LAYOUTS = frozenset({"RGGB", "GRBG", "GBRG", "BGGR"})
SUPPORTED_TORCH_DTYPES = frozenset(
    {"uint8", "uint16", "int16", "int32", "float16", "float32", "float64"}
)


@dataclass(frozen=True, slots=True)
class DebayerConfig:
    """Pure-Python configuration for `DebayerProgram` (does not import `torch`)."""

    height: int
    width: int
    layout_name: str = "RGGB"
    device: str = "cuda"
    input_dtype: str = "uint8"
    output_dtype: str = "uint8"
    in_maxval: float = 255.0
    out_maxval: float = 255.0
    round_decimal: int = 0
    min_val: float = 0.0
    max_val: float = 255.0
    num_streams: int = 2

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0 or self.height % 2 != 0 or self.width % 2 != 0:
            raise ValueError(
                f"height and width must be positive even integers, got ({self.height}, {self.width})"
            )
        if self.layout_name not in VALID_LAYOUTS:
            raise ValueError(
                f"Invalid layout_name {self.layout_name!r}; must be one of {sorted(VALID_LAYOUTS)}"
            )
        if self.in_maxval <= 0:
            raise ValueError(f"in_maxval must be positive, got {self.in_maxval}")
        if self.num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {self.num_streams}")
        if self.min_val > self.max_val:
            raise ValueError(
                f"min_val ({self.min_val}) must be <= max_val ({self.max_val})"
            )

        in_spec = BufferSpec((self.height, self.width), self.input_dtype)
        out_spec = BufferSpec((self.height, self.width, 3), self.output_dtype)
        if in_spec.dtype_str not in SUPPORTED_TORCH_DTYPES:
            raise ValueError(
                f"Unsupported input_dtype {in_spec.dtype_str!r}; "
                f"supported: {sorted(SUPPORTED_TORCH_DTYPES)}"
            )
        if out_spec.dtype_str not in SUPPORTED_TORCH_DTYPES:
            raise ValueError(
                f"Unsupported output_dtype {out_spec.dtype_str!r}; "
                f"supported: {sorted(SUPPORTED_TORCH_DTYPES)}"
            )
        object.__setattr__(self, "input_dtype", in_spec.dtype_str)
        object.__setattr__(self, "output_dtype", out_spec.dtype_str)

    @property
    def input_spec(self) -> BufferSpec:
        return BufferSpec((self.height, self.width), self.input_dtype)

    @property
    def output_spec(self) -> BufferSpec:
        return BufferSpec((self.height, self.width, 3), self.output_dtype)


@dataclass(frozen=True, slots=True)
class DebayerProgramFactory:
    """Picklable, torch-free factory exposing buffer specs and creating `DebayerProgram` in worker."""

    config: DebayerConfig

    @property
    def input_spec(self) -> BufferSpec:
        return self.config.input_spec

    @property
    def output_spec(self) -> BufferSpec:
        return self.config.output_spec

    def __call__(self) -> "IGpuProgram":
        from zero_copy_pipe.programs.debayer_program import DebayerProgram

        return DebayerProgram(self.config)
