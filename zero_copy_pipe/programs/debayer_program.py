from dataclasses import dataclass
from typing import List, Optional

import torch

from zero_copy_pipe.dto import BufferSpec
from zero_copy_pipe.interfaces import IGpuProgram
from zero_copy_pipe.programs.debayer_5x5 import Debayer5x5
from zero_copy_pipe.programs.debayer_config import DebayerConfig


@dataclass(slots=True)
class _StreamSlot:
    stream: Optional[torch.cuda.Stream]
    h2d_done: Optional[torch.cuda.Event]
    d2h_done: Optional[torch.cuda.Event]
    gpu_in: torch.Tensor
    gpu_in_f32: torch.Tensor
    gpu_out_f32: torch.Tensor
    gpu_out: torch.Tensor


class DebayerProgram(IGpuProgram):
    """Pre-allocated Malvar-He-Cutler 5x5 debayering stage supporting multi-stream CUDA overlap."""

    def __init__(self, config: DebayerConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._input_spec = config.input_spec
        self._output_spec = config.output_spec
        self._in_torch_dtype = self._input_spec.torch_dtype()
        self._out_torch_dtype = self._output_spec.torch_dtype()
        self._module: Optional[Debayer5x5] = None
        self._slots: List[_StreamSlot] = []

    @property
    def config(self) -> DebayerConfig:
        return self._config

    @property
    def input_spec(self) -> BufferSpec:
        return self._input_spec

    @property
    def output_spec(self) -> BufferSpec:
        return self._output_spec

    @property
    def uses_cuda(self) -> bool:
        return self._device.type == "cuda"

    @property
    def num_streams(self) -> int:
        return self._config.num_streams if self.uses_cuda else 1

    def setup(self) -> None:
        if self._module is not None or self._slots:
            raise RuntimeError("DebayerProgram is already initialized; call close() first")
        if self.uses_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")

        try:
            module = Debayer5x5(
                height=self._config.height,
                width=self._config.width,
                layout_name=self._config.layout_name,
                device=self._device,
            )
            module.eval()
            self._module = module

            in_shape = self._input_spec.shape
            out_shape = self._output_spec.shape
            for _ in range(self.num_streams):
                stream = torch.cuda.Stream(device=self._device) if self.uses_cuda else None
                h2d_done = torch.cuda.Event() if self.uses_cuda else None
                d2h_done = torch.cuda.Event() if self.uses_cuda else None

                gpu_in_f32 = torch.empty(
                    (1, 1, self._config.height, self._config.width),
                    dtype=torch.float32,
                    device=self._device,
                )
                if self._in_torch_dtype == torch.float32:
                    gpu_in = gpu_in_f32[0, 0]
                else:
                    gpu_in = torch.empty(
                        in_shape, dtype=self._in_torch_dtype, device=self._device
                    )

                gpu_out_f32 = torch.empty(
                    out_shape, dtype=torch.float32, device=self._device
                )
                if self._out_torch_dtype == torch.float32:
                    gpu_out = gpu_out_f32
                else:
                    gpu_out = torch.empty(
                        out_shape, dtype=self._out_torch_dtype, device=self._device
                    )

                self._slots.append(
                    _StreamSlot(
                        stream=stream,
                        h2d_done=h2d_done,
                        d2h_done=d2h_done,
                        gpu_in=gpu_in,
                        gpu_in_f32=gpu_in_f32,
                        gpu_out_f32=gpu_out_f32,
                        gpu_out=gpu_out,
                    )
                )
        except BaseException:
            self.close()
            raise

    def warmup(self) -> None:
        if self._module is None or not self._slots:
            raise RuntimeError("Call setup() before warmup()")

        host_in = torch.zeros(
            self._input_spec.shape,
            dtype=self._in_torch_dtype,
            pin_memory=self.uses_cuda,
        )
        host_out = torch.zeros(
            self._output_spec.shape,
            dtype=self._out_torch_dtype,
            pin_memory=self.uses_cuda,
        )
        for slot_idx in range(len(self._slots)):
            _, done_ev = self.enqueue(host_in, host_out, stream_slot=slot_idx)
            if done_ev is not None:
                done_ev.synchronize()

    def enqueue(
        self,
        host_input: torch.Tensor,
        host_output: torch.Tensor,
        stream_slot: int = 0,
    ) -> tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]:
        if self._module is None or not self._slots:
            raise RuntimeError("Program is not initialized; call setup() first")
        if not (0 <= stream_slot < len(self._slots)):
            raise ValueError(
                f"stream_slot {stream_slot} out of range [0, {len(self._slots)})"
            )
        if tuple(host_input.shape) != self._input_spec.shape:
            raise ValueError(
                f"Expected input shape {self._input_spec.shape}, got {tuple(host_input.shape)}"
            )
        if tuple(host_output.shape) != self._output_spec.shape:
            raise ValueError(
                f"Expected output shape {self._output_spec.shape}, got {tuple(host_output.shape)}"
            )
        if host_input.dtype != self._in_torch_dtype:
            raise ValueError(
                f"Expected input dtype {self._in_torch_dtype}, got {host_input.dtype}"
            )
        if host_output.dtype != self._out_torch_dtype:
            raise ValueError(
                f"Expected output dtype {self._out_torch_dtype}, got {host_output.dtype}"
            )
        if host_input.device.type != "cpu":
            raise ValueError(
                f"Expected CPU host_input tensor, got device {host_input.device}"
            )
        if host_output.device.type != "cpu":
            raise ValueError(
                f"Expected CPU host_output tensor, got device {host_output.device}"
            )

        slot = self._slots[stream_slot]
        if slot.stream is not None:
            with torch.cuda.stream(slot.stream):
                slot.gpu_in.copy_(host_input, non_blocking=True)
                assert slot.h2d_done is not None
                slot.h2d_done.record(slot.stream)
                self._execute_slot(slot, host_output)
                assert slot.d2h_done is not None
                slot.d2h_done.record(slot.stream)
            return slot.h2d_done, slot.d2h_done

        slot.gpu_in.copy_(host_input)
        self._execute_slot(slot, host_output)
        return None, None

    def _execute_slot(self, slot: _StreamSlot, host_output: torch.Tensor) -> None:
        assert self._module is not None
        if self._in_torch_dtype != torch.float32:
            slot.gpu_in_f32[0, 0].copy_(slot.gpu_in)

        with torch.no_grad():
            self._module(
                slot.gpu_in_f32,
                slot.gpu_out_f32,
                self._config.in_maxval,
                self._config.out_maxval,
                self._config.round_decimal,
                self._config.min_val,
                self._config.max_val,
            )

        if self._out_torch_dtype != torch.float32:
            slot.gpu_out.copy_(slot.gpu_out_f32)

        host_output.copy_(slot.gpu_out, non_blocking=self.uses_cuda)

    def close(self) -> None:
        for slot in self._slots:
            if slot.stream is not None:
                try:
                    slot.stream.synchronize()
                except Exception:
                    pass
        self._slots.clear()
        self._module = None
