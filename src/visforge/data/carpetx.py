"""Discovery for CarpetX/SimFactory output trees."""

from __future__ import annotations

from pathlib import Path

from visforge.data.detect import (
    detect_backend,
    is_silo_fragment,
    parse_iteration,
    parse_plane,
    parse_tsv_name,
)
from visforge.data.model import DataFile, DatasetIndex


def discover(path: str | Path) -> DatasetIndex:
    """Discover supported data artifacts below a CarpetX output path.

    ``path`` may be a simulation directory, an ``output-0000/<simulation>``
    directory, a single `.bp5` directory, a `.silo` file, or a TSV line file.
    """

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    data_files: list[DataFile] = []
    manifests: list[Path] = []
    metadata: list[Path] = []

    candidates = _iter_candidates(root)
    for candidate in candidates:
        if candidate.suffix in {".yaml", ".yml"} and "metadata" in candidate.name:
            metadata.append(candidate)
            continue
        if candidate.suffix == ".visit":
            manifests.append(candidate)
            continue

        backend = detect_backend(candidate)
        if backend is None:
            continue
        if backend == "silo" and is_silo_fragment(candidate):
            continue
        if backend == "tsv":
            parsed = parse_tsv_name(candidate)
            if parsed is None:
                continue
            variable, iteration, axis = parsed
            data_files.append(
                DataFile(
                    path=candidate,
                    backend="tsv",
                    iteration=iteration,
                    axis=axis,
                    variable=variable,
                )
            )
            continue

        data_files.append(
            DataFile(
                path=candidate,
                backend=backend,
                iteration=parse_iteration(candidate),
                plane=parse_plane(candidate),
            )
        )

    return DatasetIndex(
        root=root,
        files=tuple(sorted(data_files, key=lambda file: str(file.path))),
        manifests=tuple(sorted(manifests)),
        metadata=tuple(sorted(metadata)),
    )


def _iter_candidates(root: Path) -> tuple[Path, ...]:
    if root.is_file():
        return (root,)
    if root.name.endswith(".bp5"):
        return (root,)

    candidates: list[Path] = []
    for pattern in ("*.bp5", "*.silo", "*.tsv", "*.visit", "*.yaml", "*.yml"):
        candidates.extend(root.rglob(pattern))
    return tuple(candidates)
