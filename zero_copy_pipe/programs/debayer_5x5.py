# MIT License

# Copyright (c) 2020 Christoph Heindl
# Modifications Copyright (c) 2026 Patrick Joel Tirta

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from https://github.com/cheind/pytorch-debayer

import enum

import torch


class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0, G=1, B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)


class Debayer5x5(torch.nn.Module):
    """Demosaicing of Bayer images using the Malvar-He-Cutler 5x5 algorithm."""

    def __init__(
        self,
        height: int,
        width: int,
        layout_name: str = "RGGB",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if height <= 0 or width <= 0 or height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                f"Height and width must be positive even integers, got ({height}, {width})"
            )
        try:
            layout_enum = Layout[layout_name]
        except KeyError as exc:
            raise ValueError(
                f"Invalid layout name: {layout_name}. Must be one of {[layout.name for layout in Layout]}"
            ) from exc

        self.height = height
        self.width = width
        self.layout_name = layout_name

        kernels = (
            torch.tensor(
                [
                    # G at R,B locations (scaled by 16)
                    [0, 0, -2, 0, 0],
                    [0, 0, 4, 0, 0],
                    [-2, 4, 8, 4, -2],
                    [0, 0, 4, 0, 0],
                    [0, 0, -2, 0, 0],
                    # R,B at G in R rows (scaled by 16)
                    [0, 0, 1, 0, 0],
                    [0, -2, 0, -2, 0],
                    [-2, 8, 10, 8, -2],
                    [0, -2, 0, -2, 0],
                    [0, 0, 1, 0, 0],
                    # R,B at G in B rows (scaled by 16)
                    [0, 0, -2, 0, 0],
                    [0, -2, 8, -2, 0],
                    [1, 0, 10, 0, 1],
                    [0, -2, 8, -2, 0],
                    [0, 0, -2, 0, 0],
                    # R at B and B at R (scaled by 16)
                    [0, 0, -3, 0, 0],
                    [0, 4, 0, 4, 0],
                    [-3, 0, 12, 0, -3],
                    [0, 4, 0, 4, 0],
                    [0, 0, -3, 0, 0],
                ],
                dtype=torch.float32,
                device=device,
            ).view(4, 1, 5, 5)
            / 16.0
        )
        self.kernels = torch.nn.Buffer(kernels)

        # Precompute direct 2x2 BGR scatter schedule (eliminates int64 full-frame index,
        # torch.cat, torch.gather, .flip(0), and .permute().contiguous()).
        bgr_tile = self._index_from_layout(layout_enum).squeeze(0).flip(0)
        self._scatter_plan: tuple[tuple[int, int, int, int], ...] = tuple(
            (out_ch, row_off, col_off, int(bgr_tile[out_ch, row_off, col_off].item()))
            for out_ch in range(3)
            for row_off in range(2)
            for col_off in range(2)
        )

    def forward(
        self,
        x: torch.Tensor,
        out_f32: torch.Tensor,
        in_maxval: float,
        out_maxval: float,
        round_decimal: int,
        min_val: float,
        max_val: float,
    ) -> None:
        """Debayer 1x1xHxW float32 tensor `x` in-place into pre-allocated HxWx3 float32 `out_f32` (BGR)."""
        x.mul_(1.0 / in_maxval).clamp_(0.0, 1.0)
        xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
        planes = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)

        x_hw = x[0, 0]
        planes_chw = planes[0]
        for out_ch, row_off, col_off, src_idx in self._scatter_plan:
            src_plane = x_hw if src_idx == 4 else planes_chw[src_idx]
            out_f32[row_off::2, col_off::2, out_ch].copy_(src_plane[row_off::2, col_off::2])

        out_f32.clamp_(0.0, 1.0).mul_(out_maxval).round_(decimals=round_decimal).clamp_(
            min=min_val, max=max_val
        )

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        rggb = torch.tensor(
            [
                # R channel (R, G1; G2, B)
                [4, 1],
                [2, 3],
                # G channel
                [0, 4],
                [4, 0],
                # B channel
                [3, 2],
                [1, 4],
            ]
        ).view(1, 3, 2, 2)
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }[layout]
