from __future__ import annotations

from pathlib import Path

import numpy as np

from visforge.data.model import FieldInfo, LineData
from visforge.plotting.line import plot_line


def test_plot_line_writes_png(tmp_path: Path) -> None:
    line = LineData(
        field=FieldInfo(name="rho"),
        iteration=0,
        time=0.0,
        axis="x",
        coordinate=np.array([0.0, 1.0, 2.0]),
        values=np.array([1.0, 3.0, 2.0]),
    )
    output = tmp_path / "line.png"
    result = plot_line(line, output=output)
    assert result.output == output.resolve()
    assert output.stat().st_size > 0
