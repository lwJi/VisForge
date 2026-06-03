from __future__ import annotations

from pathlib import Path

import numpy as np

from visforge.data.model import FieldInfo, GridBlock, SliceData
from visforge.plotting.scalar import plot_scalar_slice


def test_plot_scalar_slice_writes_png(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(9, dtype=float).reshape(3, 3),
        axes=("y", "x"),
        origin=(0.0, 0.0),
        spacing=(0.5, 0.5),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )
    output = tmp_path / "slice.png"
    result = plot_scalar_slice(slice_data, output=output)
    assert result.output == output.resolve()
    assert output.stat().st_size > 0


def test_plot_scalar_slice_can_overlay_mesh(tmp_path: Path) -> None:
    block = GridBlock(
        data=np.arange(4, dtype=float).reshape(2, 2),
        axes=("y", "x"),
        origin=(-1.0, 2.0),
        spacing=(0.5, 0.25),
    )
    slice_data = SliceData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        plane="xy",
        blocks=(block,),
    )
    output = tmp_path / "slice_mesh.png"
    result = plot_scalar_slice(slice_data, output=output, show_mesh=True)
    assert result.output == output.resolve()
    assert output.stat().st_size > 0
    assert len(result.axes.patches) == 1
