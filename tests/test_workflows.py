from __future__ import annotations

from pathlib import Path

import pytest

from visforge.workflows.inspect import format_summary, inspect_dataset
from visforge.workflows.line_plot import make_line_plot

pytestmark = pytest.mark.integration


def test_inspect_example_data(example_data_root) -> None:
    summary = inspect_dataset(example_data_root / "testoutput2d", backend="tsv")
    assert "tsv" in summary.backends
    assert "gfc" in summary.fields
    assert "gfv" in summary.fields
    assert "Backends:" in format_summary(summary)


def test_line_workflow_example_data(example_data_root, tmp_path: Path) -> None:
    output = tmp_path / "gfc_x.png"
    result = make_line_plot(
        example_data_root / "testoutput2d",
        field="gfc",
        axis="x",
        output=output,
    )
    assert result.output == output.resolve()
    assert output.stat().st_size > 0
