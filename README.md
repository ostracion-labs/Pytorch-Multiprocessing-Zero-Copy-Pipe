# Zero-Copy Multiprocess Image Processing Pipe (`zero_copy_pipe`)

A generic, modular 3-end zero-copy multiprocess pipeline in Python combining a contiguous page-aligned `multiprocessing.shared_memory` arena, `cudaHostRegister` DMA pinning, configurable dual-stream CUDA execution (`num_streams=2`), 128-bit stale-group cancellation, and PyTorch-free downstream CPU workers.

It includes a built-in `DebayerProgram` implementing Malvar-He-Cutler 5×5 debayering (supporting `uint8` and 10/12/16-bit `uint16` Bayer inputs across `RGGB`, `GRBG`, `GBRG`, and `BGGR`), while letting you swap any of the three ends independently.

---

## Architecture: The 3-End Pipe

```text
  END 1: Input (Producer)             zero_copy_pipe Core                   END 3: Output (Consumers)
┌─────────────────────────┐    ┌───────────────────────────────────────┐    ┌───────────────────────────┐
│ User / Camera SDK       │    │  Input SharedMemoryArena              │    │ N CPU Worker Processes    │
│                         │    │   • 1 contiguous mmap, 4096B-aligned  │    │   • spawn, torch-free     │
│ with pipe.acquire_input │───▶│   • in_work_q (PacketDto) / in_free_q │    │   • read-only SHM views   │
│   () as slot:           │    │                  ▼                    │───▶│   • IOutputProcessor      │
│   grab_into(slot.buf)   │    │  END 2: GPU Worker(s) (spawn)         │    │     (setup once/worker)   │
│                         │    │   • 1x cudaHostRegister per arena     │    └─────────────┬─────────────┘
│ (or pipe.submit(arr))   │    │   • IGpuProgram (default: Debayer5x5) │                  │
└─────────────────────────┘    │   • num_streams=2 overlap + early H2D │      result_q (optional)
                               │     input-slot release                │                  ▼
                               │   • 128-bit ActiveGroupTracker        │           pipe.get_result()
                               │  Output SharedMemoryArena             │
                               └───────────────────────────────────────┘
```

### Core Components
- **Zero-Copy Contracts (`zero_copy_pipe/dto/`):**
  - `BufferSpec(shape, dtype_str)`: pure-Python slot shape & dtype contract (no `torch` import in parent or CPU workers).
  - `PacketDto(group_id, group_index, dimension, data_id)`: `@dataclass(frozen=True, slots=True)` lease metadata supporting full 128-bit `group_id`s (e.g., `uuid.uuid4().int`).
- **Pluggable End Contracts (`zero_copy_pipe/interfaces/`):**
  - `IGpuProgram`: swappable GPU/compute stage with `input_spec`, `output_spec`, `uses_cuda`, `num_streams`, `setup()`, `warmup()`, `enqueue(..., stream_slot)`, and `close()`.
  - `IOutputProcessor`: swappable stateful CPU stage with `setup()`, `process(frame, packet)`, and `close()` (plain callables are auto-wrapped via `CallableOutputProcessorFactory`).
- **Contiguous `SharedMemoryArena` (`zero_copy_pipe/pipeline/shm_arena.py`):**
  - One POSIX `SharedMemory` segment per pool partitioned at 4096-byte page boundaries—one fd, one `mmap`, one `cudaHostRegister` / `cudaHostUnregister`, and transactional rollback on failure.
  - GPU workers map slots via `torch.frombuffer(shm.buf, dtype=..., count=spec.numel, offset=slot_offset).view(spec.shape)`.
  - CPU workers open handles with `track=False` and expose **read-only** NumPy arrays (`writeable = False`).
- **Real-Time Stale-Group Cancellation (`ActiveGroupTracker`):**
  - `pipe.set_active_group(group_id)` / `pipe.clear_active_group()` writes a 128-bit group ID into shared synchronized `uint64`s so GPU and CPU workers drop outdated frames immediately and recycle slots to recover from lag.

---

## Quickstart

### 1. Built-In 5×5 Debayer Pipe (Zero-Copy Input)

```python
import uuid
import numpy as np
from zero_copy_pipe import DebayerConfig, DebayerProgramFactory, PipeManager

def analyze_frame(bgr_frame: np.ndarray, pkt) -> tuple[int, tuple[float, ...]]:
    # `bgr_frame` is a read-only zero-copy NumPy view over Output SharedMemoryArena
    mean_bgr = tuple(float(bgr_frame[:, :, c].mean()) for c in range(3))
    return pkt.group_index, mean_bgr

cfg = DebayerConfig(
    height=1080,
    width=1920,
    layout_name="RGGB",
    device="cuda",
    input_dtype="uint16",   # e.g., 10-bit camera Bayer sensor
    output_dtype="uint8",
    in_maxval=1023.0,
    out_maxval=255.0,
    num_streams=2,          # Dual CUDA stream overlap
)

batch_id = uuid.uuid4().int

with PipeManager(
    gpu_program_factory=DebayerProgramFactory(cfg),
    output_processor=analyze_frame,
    num_input_slots=4,
    num_output_slots=4,
    num_cpu_workers=2,
) as pipe:
    pipe.set_active_group(batch_id)

    for idx in range(10):
        with pipe.acquire_input(group_id=batch_id, group_index=idx) as slot:
            # Write directly from camera SDK or capture source into `slot.buffer`
            slot.buffer[:] = idx

    results = [pipe.get_result(timeout=10.0) for _ in range(10)]
```

### 2. Swapping End 2 (`IGpuProgram`) and End 3 (`IOutputProcessor`)

Implement `IGpuProgram` for any custom PyTorch/CUDA stage (and pass a picklable factory exposing `.input_spec` and `.output_spec`, or pass `input_spec=` / `output_spec=` directly to `PipeManager`), paired with a stateful `IOutputProcessor` factory whose `setup()` runs once per CPU worker process (for example, loading an inference session, compressor, or disk writer):

```python
from dataclasses import dataclass
import numpy as np
from zero_copy_pipe import BufferSpec, IGpuProgram, IOutputProcessor, PacketDto, PipeManager

class CustomConsumer(IOutputProcessor):
    def setup(self) -> None:
        self.count = 0  # Initialize heavy worker-local resources once

    def process(self, frame: np.ndarray, packet: PacketDto) -> int:
        self.count += 1
        return int(frame.sum())

    def close(self) -> None:
        pass

@dataclass(frozen=True, slots=True)
class CustomConsumerFactory:
    def __call__(self) -> IOutputProcessor:
        return CustomConsumer()
```

---

## Design Tradeoffs & System Constraints

- **Full-Duplex Dual Streams (`num_streams=2`) vs. VRAM Footprint:**
  PCIe provides separate TX (H2D) and RX (D2H) links, and GPUs with 2 async copy engines (e.g., `asyncEngineCount == 2` on Turing/GTX 1650) can overlap H2D, kernel compute, and D2H across two streams while also hiding Python queue dispatch latency. Because each concurrent stream requires its own pre-allocated VRAM buffers, VRAM scales linearly with `num_streams` (`num_streams=1` halves VRAM when memory is tight; consumer WDDM/WSL2 drivers may also serialize some concurrent H2D/D2H transfers, so benchmark on your target hardware).
- **PCIe Round-Trip vs. Keeping Data on GPU:**
  Malvar-He-Cutler 5×5 debayering is memory-bandwidth-bound with low arithmetic intensity. Transferring raw Bayer H2D and debayered 3-channel BGR D2H over PCIe only pays off when downstream stages are genuinely CPU-bound (encoding, disk writes, UI, multi-core CPU analytics) or when GPU stages expand beyond lightweight debayering; if downstream stages also run on GPU (e.g., segmentation), keeping tensors in VRAM avoids D2H entirely. For hybrid CPU/GPU downstream workflows (e.g. need to find the object using OpenCV and re-run segmentation on the GPU), if VRAM allows, holding the finished debayer on GPU and utilizing 2nd multiprocessing queue to pass on crop area from the CPU process to the GPU 2nd process, sharing the pre-allocated VRAM pool via CUDA IPC is an alternative to reduce one data copy pass, at the expense of needing more VRAM buffers. The CPU process should always send the ID (with None/[0,0] as crop position) to the 2nd process even on early exit or throws since that ID is needed to recycle the VRAM back. A more optimized version can run with multiple OS threads (one per stage/CUDA stream, or async via asyncio.to_thread) since the thread blockers are at `multiprocessing.Queue.get()` and `stream.synchronize()`, so only one process (and therefore CUDA context) is necessary per GPU just to handle the Queues and hand over the compute to CUDA. On the `stream.synchronize()`, blocking the OS thread here should be a non-issue since we only have one RX and one TX in default settings, as such multiprocessing can lead to oversubscription and serialization.
- **Multiprocessing vs. Threads:**
  Standard NumPy, OpenCV, and PyTorch release the Python GIL, meaning single-process `ThreadPoolExecutor` + `pin_memory=True` can outperform multiprocessing when downstream work is short and C-extension-backed. Multiprocessing (`spawn`) is retained here to isolate GIL-heavy Python post-processing, process crashes, and GC pauses across cores—while ensuring CPU workers never import `torch`. Scikit-image, however, has less C-extension backing and may run heavy computation directly in Python, which this library should help a lot (it has a `medial_axis` function that can create skeleton complete with the distance transform). Also, there needs to be further testing for multithreaded (`ThreadPoolExecutor`) downstream processing with heavy NumPy & OpenCV processing but with many if-checks that fail early, and on i7-grade CPUs that have 20+ threads.
- **`/dev/shm` & `RLIMIT_MEMLOCK` Limits:**
  POSIX `SharedMemory` backs onto `/dev/shm` (`tmpfs`), and `cudaHostRegister` locks physical pages in RAM (`RLIMIT_MEMLOCK`). Default Docker containers cap `/dev/shm` at 64 MB—insufficient for multi-slot 4K RGB arenas unless started with `--shm-size`. Contiguous `SharedMemoryArena` minimizes file descriptors and driver registrations (1 per pool), and `cudaHostRegister` behavior varies on unified-memory SoCs (e.g., Jetson) and virtualized GPUs.
- **Eager PyTorch Execution on Python 3.15:**
  On Python 3.15, both `torch.jit.script` (deprecated / warning on 3.14+) and `torch.compile` (`RuntimeError` on 3.15+) are unsupported. `Debayer5x5` uses precomputed direct 2×2 BGR slice copies instead of `int64` `torch.gather` + `flip`, cutting peak VRAM per stream by >2× (~41 bytes/px vs ~100 bytes/px at 1024×1024) in pure eager PyTorch.

---

## Running Tests

```bash
.pixi/envs/default/bin/pytest -v
# or via pixi:
pixi run test
```

*(CUDA-specific tests are automatically skipped if no CUDA device is available.)*
