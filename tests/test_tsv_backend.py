from __future__ import annotations

from pathlib import Path

from visforge.data.tsv import TsvBackend


def test_tsv_backend_reads_line(tmp_path: Path) -> None:
    path = tmp_path / "run-gfc.it000000.x.tsv"
    path.write_text(
        "# 1:iteration\t2:time\t3:patch\t4:level\t5:x\t6:gfc\n"
        "0\t0.0\t0\t0\t1.0\t2.0\n"
        "0\t0.0\t0\t0\t0.0\t1.0\n",
        encoding="utf-8",
    )

    backend = TsvBackend(tmp_path)
    assert backend.list_iterations() == (0,)
    assert [field.name for field in backend.list_fields()] == ["gfc"]

    line = backend.read_line("gfc", axis="x")
    assert line.axis == "x"
    assert line.coordinate.tolist() == [0.0, 1.0]
    assert line.values.tolist() == [1.0, 2.0]
