# VisForge

VisForge is a small, modular Python codebase for visualizing CarpetX simulation
output. The initial scope supports CarpetX output discovery, TSV line plots,
and clean backend interfaces for openPMD and Silo data.

## Development

This repository uses the Python interpreter specified in `AGENTS.md`:

```bash
$HOME/Misc/venv/py3/bin/python -m pytest
```

## Example Commands

List available commands and options:

```bash
$HOME/Misc/venv/py3/bin/visforge --help
$HOME/Misc/venv/py3/bin/visforge inspect --help
$HOME/Misc/venv/py3/bin/visforge plot-line --help
$HOME/Misc/venv/py3/bin/visforge plot-slice --help
```

Inspect a CarpetX output tree:

```bash
visforge inspect /Users/liwei/docker-workspace/data/TestOutput2D/testoutput2d
```

Plot a TSV line output:

```bash
visforge plot-line /Users/liwei/docker-workspace/data/TestOutput2D/testoutput2d \
  --field gfc \
  --axis x \
  --output gfc_x.png
```
