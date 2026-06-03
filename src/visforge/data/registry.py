"""Backend registration and selection."""

from __future__ import annotations

from pathlib import Path

from visforge.data.backend import DataBackend
from visforge.data.carpetx import discover
from visforge.data.detect import detect_backend
from visforge.data.model import BackendName
from visforge.data.openpmd import OpenPMDBackend
from visforge.data.silo import SiloBackend
from visforge.data.tsv import TsvBackend


def open_dataset(path: str | Path, backend: BackendName | str = "auto") -> DataBackend:
    """Open a dataset with the requested backend.

    With ``backend="auto"``, selection is based on the path itself for single
    files and otherwise on discovered CarpetX artifacts. TSV is preferred for
    line-only data, while openPMD is preferred for volumetric/slice data when
    both openPMD and Silo are present.
    """

    resolved = Path(path).expanduser().resolve()
    selected = _select_backend(resolved, backend)
    if selected == "tsv":
        return TsvBackend(resolved)
    if selected == "openpmd":
        return OpenPMDBackend(resolved)
    if selected == "silo":
        return SiloBackend(resolved)
    raise ValueError(f"Unsupported backend {backend!r}")


def _select_backend(path: Path, backend: BackendName | str) -> BackendName:
    if backend != "auto":
        if backend not in {"openpmd", "silo", "tsv"}:
            raise ValueError(f"Unsupported backend {backend!r}")
        return backend  # type: ignore[return-value]

    direct = detect_backend(path)
    if direct is not None:
        return direct

    index = discover(path)
    available = set(index.backends)
    if "openpmd" in available:
        return "openpmd"
    if "silo" in available:
        return "silo"
    if "tsv" in available:
        return "tsv"
    raise FileNotFoundError(f"No supported VisForge data files found under {path}")
