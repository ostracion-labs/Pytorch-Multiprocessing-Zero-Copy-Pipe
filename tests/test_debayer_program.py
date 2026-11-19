import numpy as np
import pytest
import torch

from zero_copy_pipe import Debayer5x5, DebayerConfig, DebayerProgram, Layout


def _reference_malvar_bgr(
    bayer: np.ndarray,
    layout_name: str,
    in_maxval: float = 255.0,
    out_maxval: float = 255.0,
) -> torch.Tensor:
    helper = Debayer5x5(bayer.shape[0], bayer.shape[1], layout_name=layout_name)
    idx_ref = helper._index_from_layout(Layout[layout_name]).repeat(
        1, 1, bayer.shape[0] // 2, bayer.shape[1] // 2
    )
    x = torch.from_numpy(bayer).float()[None, None].mul_(1.0 / in_maxval).clamp_(0.0, 1.0)
    xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
    planes = torch.nn.functional.conv2d(xpad, helper.kernels, stride=1)
    planes = torch.cat((planes, x), 1)
    return (
        torch.gather(planes, 1, idx_ref)
        .squeeze(0)
        .flip(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(out_maxval)
        .round()
        .clamp(0.0, 255.0)
        .to(torch.uint8)
    )


def test_debayer_config_and_module_validation() -> None:
    with pytest.raises(ValueError):
        DebayerConfig(height=7, width=8)
    with pytest.raises(ValueError):
        DebayerConfig(height=8, width=8, layout_name="INVALID")
    with pytest.raises(ValueError):
        DebayerConfig(height=8, width=8, num_streams=0)
    with pytest.raises(ValueError):
        DebayerConfig(height=8, width=8, in_maxval=0.0)
    with pytest.raises(ValueError):
        DebayerConfig(height=8, width=8, min_val=10.0, max_val=5.0)
    with pytest.raises(ValueError):
        DebayerConfig(height=8, width=8, input_dtype="uint64")
    with pytest.raises(ValueError):
        Debayer5x5(height=7, width=8, layout_name="RGGB")


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("layout_name", ["RGGB", "GRBG", "GBRG", "BGGR"])
def test_debayer_all_layouts_match_reference(device: str, layout_name: str) -> None:
    rng = np.random.default_rng(123)
    bayer = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    expected = _reference_malvar_bgr(bayer, layout_name)

    cfg = DebayerConfig(
        height=32,
        width=32,
        layout_name=layout_name,
        device=device,
        num_streams=2,
    )
    out = torch.empty((32, 32, 3), dtype=torch.uint8)
    with DebayerProgram(cfg) as prog:
        prog.run(torch.from_numpy(bayer), out)

    assert torch.equal(out, expected)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_debayer_10bit_uint16_to_uint8(device: str) -> None:
    rng = np.random.default_rng(7)
    bayer_u16 = rng.integers(0, 1024, size=(24, 24), dtype=np.uint16)
    expected = _reference_malvar_bgr(bayer_u16, "RGGB", in_maxval=1023.0, out_maxval=255.0)

    cfg = DebayerConfig(
        height=24,
        width=24,
        layout_name="RGGB",
        device=device,
        input_dtype="uint16",
        output_dtype="uint8",
        in_maxval=1023.0,
        out_maxval=255.0,
        num_streams=2,
    )
    out = torch.empty((24, 24, 3), dtype=torch.uint8)
    with DebayerProgram(cfg) as prog:
        prog.run(torch.from_numpy(bayer_u16), out)

    assert torch.equal(out, expected)


def test_debayer_program_validates_shape_and_dtype() -> None:
    cfg = DebayerConfig(height=16, width=16, device="cpu", num_streams=1)
    with DebayerProgram(cfg) as prog:
        with pytest.raises(ValueError):
            prog.run(
                torch.zeros((16, 18), dtype=torch.uint8),
                torch.empty((16, 16, 3), dtype=torch.uint8),
            )
        with pytest.raises(ValueError):
            prog.run(
                torch.zeros((16, 16), dtype=torch.float32),
                torch.empty((16, 16, 3), dtype=torch.uint8),
            )
        with pytest.raises(ValueError):
            prog.run(
                torch.zeros((16, 16), dtype=torch.uint8),
                torch.empty((16, 16, 3), dtype=torch.float32),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_debayer_program_cuda_zero_malloc_and_low_vram() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_bytes = torch.cuda.memory_allocated()

    height, width = 512, 512
    cfg = DebayerConfig(
        height=height,
        width=width,
        layout_name="RGGB",
        device="cuda",
        num_streams=2,
    )
    host_in = torch.randint(0, 255, (height, width), dtype=torch.uint8).pin_memory()
    host_out = torch.empty((height, width, 3), dtype=torch.uint8).pin_memory()

    with DebayerProgram(cfg) as prog:
        torch.cuda.reset_accumulated_memory_stats()
        events = []
        for idx in range(6):
            _, done_ev = prog.enqueue(host_in, host_out, stream_slot=idx % prog.num_streams)
            events.append(done_ev)
        for ev in events:
            assert ev is not None
            ev.synchronize()

        stats = torch.cuda.memory_stats()
        assert stats["segment.all.allocated"] == 0

        peak_delta = torch.cuda.max_memory_allocated() - base_bytes
        bytes_per_px_per_stream = peak_delta / (height * width * cfg.num_streams)
        assert bytes_per_px_per_stream < 50.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_debayer_program_use_buffer_and_kernel() -> None:
    module = Debayer5x5(height=8, width=8, layout_name="RGGB", device=torch.device("cpu"))

    buffers = dict(module.named_buffers())
    assert "kernels" in buffers
    assert buffers["kernels"] is module.kernels
    assert "kernels" not in dict(module.named_parameters())
    assert "kernels" in module.state_dict()  # remove if using Buffer(..., persistent=False)

    # Behavioral proof: a plain untracked attribute will NOT move on .to()
    module.to(device="cuda", dtype=torch.float16)
    assert module.kernels.device.type == "cuda"
    assert module.kernels.dtype == torch.float16