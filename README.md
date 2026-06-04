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
$HOME/Misc/venv/py3/bin/visforge inspect /Users/liwei/docker-workspace/data/TestOutput2D/testoutput2d
```

Plot a TSV line output:

```bash
$HOME/Misc/venv/py3/bin/visforge plot-line /Users/liwei/docker-workspace/data/TestOutput2D/testoutput2d \
  --field gfc \
  --axis x \
  --output gfc_x.png
```

Plot an openPMD scalar slice and overlay mesh/block outlines:

```bash
$HOME/Misc/venv/py3/bin/visforge plot-slice /Users/liwei/docker-workspace/data/TestOutput2D/testoutput2d \
  --field gfc \
  --iteration 0 \
  --plane xy \
  --backend openpmd \
  --show-mesh \
  --mesh-alpha 0.75 \
  --xlim -4 4 \
  --ylim -4 4 \
  --output gfc_xy_mesh.png
```

Use a YAML config file instead of passing every option on the command line:

```bash
$HOME/Misc/venv/py3/bin/visforge plot-slice --config examples/plot_slice.yaml
```

Sample a 3D openPMD field onto a user-defined 2D plane:

```bash
$HOME/Misc/venv/py3/bin/visforge plot-slice /Users/liwei/docker-workspace/data/TestOutput3D/testoutput3d \
  --field gfc \
  --iteration 0 \
  --backend openpmd \
  --sample-plane-origin 0 0 0 \
  --sample-plane-normal 1 1 0 \
  --sample-plane-up 0 0 1 \
  --sample-plane-size 8 8 \
  --sample-plane-resolution 512 512 \
  --interpolation linear \
  --output gfc_sample_plane.png
```

CLI options override values from the config file:

```bash
$HOME/Misc/venv/py3/bin/visforge plot-slice \
  --config examples/plot_slice.yaml \
  --xlim -2 2 \
  --ylim -2 2 \
  --output gfc_zoom.png
```
