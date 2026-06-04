from __future__ import annotations

from pathlib import Path

from visforge.data.model import DatasetSummary
from visforge.workflows.inspect import format_summary


def test_format_summary_shows_field_components() -> None:
    summary = DatasetSummary(
        root=Path("/data/gaussian"),
        backends=("openpmd",),
        iterations=(384,),
        planes=("xy",),
        axes=(),
        fields=("error", "state"),
        file_count=1,
        metadata_count=0,
        field_components={
            "error": ("testsubcyclingmc2_rho_err", "testsubcyclingmc2_u_err"),
            "state": ("testsubcyclingmc2_rho", "testsubcyclingmc2_u"),
        },
    )

    text = format_summary(summary)

    assert "Fields:\n" in text
    assert "  state:" in text
    assert "    components: testsubcyclingmc2_rho, testsubcyclingmc2_u" in text
