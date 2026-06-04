# VisForge

## Python

Always use `$HOME/Misc/venv/py3/bin/python` for Python commands in this repository.

## CLI

Run VisForge through the repository Python environment:

```bash
$HOME/Misc/venv/py3/bin/visforge
```

## Tests

Run tests through the repository Python environment:

```bash
$HOME/Misc/venv/py3/bin/python -m pytest
```

Run integration tests that require the local CarpetX sample data with:

```bash
VISFORGE_TEST_DATA=/Users/liwei/docker-workspace/data/TestOutput2D \
  $HOME/Misc/venv/py3/bin/python -m pytest
```
