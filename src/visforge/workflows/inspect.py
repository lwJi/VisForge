"""Dataset inspection workflow."""

from __future__ import annotations

from pathlib import Path

from visforge.data.carpetx import discover
from visforge.data.model import DatasetSummary
from visforge.data.registry import open_dataset


def inspect_dataset(path: str | Path, *, backend: str = "auto") -> DatasetSummary:
    index = discover(path)
    fields = set(_fields_from_index(index))
    field_components: dict[str, tuple[str, ...]] = {}
    try:
        dataset = open_dataset(path, backend=backend)
        iterations = dataset.list_iterations()
        selected_iteration = max(iterations) if iterations else None
        for field in dataset.list_fields(selected_iteration):
            fields.add(field.name)
            if field.components:
                field_components[field.name] = field.components
    except Exception:
        pass

    return DatasetSummary(
        root=index.root,
        backends=tuple(index.backends),
        iterations=index.iterations,
        planes=index.planes,
        axes=index.axes,
        fields=tuple(sorted(fields)),
        file_count=len(index.files),
        metadata_count=len(index.metadata),
        field_components=field_components,
    )


def format_summary(summary: DatasetSummary) -> str:
    fields = _format_fields(summary)
    lines = [
        f"Root: {summary.root}",
        f"Supported files: {summary.file_count}",
        f"Backends: {_format_tuple(summary.backends)}",
        f"Iterations: {_format_iterations(summary.iterations)}",
        f"Planes: {_format_tuple(summary.planes)}",
        f"Line axes: {_format_tuple(summary.axes)}",
        "Fields:" + (fields if fields.startswith("\n") else f" {fields}"),
        f"Metadata files: {summary.metadata_count}",
    ]
    return "\n".join(lines)


def _fields_from_index(index) -> tuple[str, ...]:
    return tuple(sorted({file.variable for file in index.files if file.variable is not None}))


def _format_tuple(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"


def _format_fields(summary: DatasetSummary) -> str:
    if not summary.fields:
        return "none"
    if not any(summary.field_components.values()):
        return _format_tuple(summary.fields)

    lines = []
    for name in summary.fields:
        components = summary.field_components.get(name, ())
        if components:
            lines.append(f"  {name}:")
            lines.append(f"    components: {_format_tuple(components)}")
        else:
            lines.append(f"  {name}")
    return "\n" + "\n".join(lines)


def _format_iterations(values: tuple[int, ...]) -> str:
    if not values:
        return "none"
    if len(values) <= 8:
        return ", ".join(str(value) for value in values)
    return f"{values[0]} ... {values[-1]} ({len(values)} total)"
