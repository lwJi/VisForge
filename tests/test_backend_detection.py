from __future__ import annotations

from pathlib import Path

from visforge.data.detect import detect_backend, parse_iteration, parse_plane, parse_tsv_name


def test_detect_backend_from_suffixes() -> None:
    assert detect_backend(Path("run.it00000000.bp5")) == "openpmd"
    assert detect_backend(Path("run.it00000000.silo")) == "silo"
    assert detect_backend(Path("run-gfc.it000000.x.tsv")) == "tsv"


def test_parse_carpetx_names() -> None:
    assert parse_iteration(Path("test.xy.it00000042.bp5")) == 42
    assert parse_plane(Path("test.xy.it00000042.bp5")) == "xy"
    assert parse_tsv_name(Path("test-gfc.it000000.x.tsv")) == ("gfc", 0, "x")
