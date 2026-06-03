from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def example_data_root() -> Path:
    value = os.environ.get("VISFORGE_TEST_DATA")
    if not value:
        pytest.skip("VISFORGE_TEST_DATA is not set")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"VISFORGE_TEST_DATA does not exist: {path}")
    return path
