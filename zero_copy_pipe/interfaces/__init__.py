from .gpu_program import IGpuProgram
from .output_processor import (
    CallableOutputProcessor,
    CallableOutputProcessorFactory,
    IOutputProcessor,
)

__all__ = [
    "CallableOutputProcessor",
    "CallableOutputProcessorFactory",
    "IGpuProgram",
    "IOutputProcessor",
]
