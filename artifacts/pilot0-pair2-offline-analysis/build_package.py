"""Execute the three local analyses and package their descriptive notebook/report."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import sys


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]


def cell(kind, text, number):
    row = {"cell_type": kind, "id": f"pair2-{number:02d}",
           "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        row.update(execution_count=None, outputs=[])
    return row


def main():
    intro = """# Pair-2 offline analysis package

**Descriptive and post-hoc. No gate input and no design authority.**
Run `pilot0-pair2-seed29-20260830T204211Z`, seed 29; selected Stage-A
B-S@40 and B-G@20. Only existing pair-2 records and the 34 local adapter
archives are read. No remote calls, new sampling, or pair-2 inspection/control.

The package contains (1) paired item-bootstrap intervals, (2) F2 failure/length
trajectories, (3) early-window F1 statistics, (4) per-layer and cadence adapter
geometry, (5) baseline item-set overlap, and (6) separately labelled non-binding
confirmatory notes. Rate/score definitions, denominator counts, uncertainty
assumptions, and exact source paths are retained in each section.

Code cells were executed sequentially with the repository Python interpreter,
not through a Jupyter kernel service. Execution counts and stdout are saved.
The adjacent Markdown cells are the report snapshot from that execution.
Rerunning a code cell rebuilds its JSON/Markdown; rerun `build_package.py` to
refresh this notebook's saved Markdown snapshots and the combined report.

Dependencies are the repository's existing NumPy, PyTorch and safetensors;
no package was installed. Geometry execution reads about 12.9 GB of local
adapter files and uses one CPU thread for the selected tensor spot checks.
"""
    cells = [cell("markdown", intro, 0)]
    setup = '''from pathlib import Path
import json
import os
import subprocess
import sys

# Find this package whether opened at the repo root or inside the package.
here = Path.cwd().resolve()
repo = next(p for p in (here, *here.parents) if (p / "duraseed_pilot_config.yaml").exists())
package = repo / "artifacts/pilot0-pair2-offline-analysis"
python = repo / ".venv/bin/python"
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

def execute_local(script):
    result = subprocess.run([str(python), str(package / script)], cwd=repo,
                            env=env, capture_output=True, text=True, check=True)
    if result.stderr:
        print(result.stderr)
    print(f"{script}: completed locally, exit {result.returncode}")

print("Pair-2 artifact paths only; no remote clients or new samples.")
'''
    cells.append(cell("code", setup, 1))
    work = [
        ("uncertainty.py", "uncertainty.md", "uncertainty.json",
         'print(json.dumps({"contrasts": len(data["contrasts"]), "bootstrap": data["bootstrap"], "early_window_rows": len(data["early_window"])}, indent=2))'),
        ("f2_diagnostics.py", "f2_diagnostics.md", "f2_diagnostics.json",
         'print(json.dumps({"checks": data["checks"], "baseline_solved_items": data["baseline_concentration"]["baseline_solved_item_count"]}, indent=2))'),
        ("adapter_geometry.py", "geometry.md", "geometry.json",
         'print(json.dumps({k: v for k, v in data["audit"].items() if k != "checks"}, indent=2))'),
    ]
    namespace = {"__name__": "__main__"}
    execution_count = 0

    def execute(row):
        nonlocal execution_count
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile("".join(row["source"]), "pair2-notebook-cell", "exec"), namespace)
        execution_count += 1
        row["execution_count"] = execution_count
        for name, stream in (("stdout", stdout), ("stderr", stderr)):
            if stream.getvalue():
                row["outputs"].append({"output_type": "stream", "name": name,
                                       "text": stream.getvalue().splitlines(keepends=True)})
        print(f"Executed notebook cell {execution_count}", flush=True)

    execute(cells[1])
    sections = []
    for script, markdown, data_file, summary in work:
        row = cell("code", f'execute_local("{script}")\ndata = json.loads((package / "{data_file}").read_text())\n{summary}\n', len(cells))
        execute(row)
        cells.append(row)
        text = (PACKAGE / markdown).read_text()
        cells.append(cell("markdown", text, len(cells)))
        sections.append(re.sub(r"^(#{1,5}) ", r"\1# ", text, flags=re.M))
    notes_path = REPO / "docs/confirmatory-design-notes.md"
    notes = notes_path.read_text()
    notes_snapshot = notes.replace("../artifacts/pilot0-pair2-offline-analysis/README.md", "README.md")
    notes_snapshot = notes_snapshot.replace("../artifacts/pilot0-pair1-offline-analysis/", "../pilot0-pair1-offline-analysis/")
    cells.append(cell("markdown", notes_snapshot + "\nSource: [notes file](../../docs/confirmatory-design-notes.md).\n", len(cells)))
    access = '''# Full cadence/per-layer spectra remain in the sidecar, not duplicated into this notebook.
geometry = json.loads((package / "geometry.json").read_text())
selected = {r["method"]: r for r in geometry["checkpoints"] if r["selected"]}
layer_id = "0"  # Change to any layer "0".."31" or "unassigned" for local inspection.
for method in ("B-S", "B-G"):
    layer = next(r for r in selected[method]["layers"] if r["layer"] == layer_id)
    print(method, "selected update", selected[method]["step"], "layer", layer_id)
    print(json.dumps({k: v for k, v in layer.items() if k != "block_diagonal_ba_singular_values"}, indent=2))
print("Complete ordered spectra: geometry['checkpoints'][...]['layers'][...]['block_diagonal_ba_singular_values']")
print("Exact baseline and overlap item IDs: f2_diagnostics.json / baseline_concentration")
'''
    row = cell("code", access, len(cells))
    execute(row)
    cells.append(row)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3 (DuraSeed local)", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": sys.version.split()[0]},
                     "execution": {"engine": "sequential standard-Python exec with captured stdout", "remote_calls": False,
                                   "code_cells_executed": execution_count, "analysis_role": "post_hoc_descriptive_no_gate_use"}}}
    (PACKAGE / "pair2-offline-analysis.ipynb").write_text(json.dumps(notebook, indent=2, allow_nan=False)+"\n")
    navigation = """
## Files and exact data

- [Executed descriptive notebook](pair2-offline-analysis.ipynb)
- [Uncertainty and early-window F1](uncertainty.md) · [exact JSON](uncertainty.json)
- [F2 diagnostics and baseline items](f2_diagnostics.md) · [exact JSON](f2_diagnostics.json)
- [Adapter geometry](geometry.md) · [all module/layer/cadence spectra](geometry.json)
- [Non-binding confirmatory notes](../../docs/confirmatory-design-notes.md)

The three analysis scripts and `build_package.py` are included beside these
outputs. From the repository root, reproduce with:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python artifacts/pilot0-pair2-offline-analysis/build_package.py
```

All 14 contrast intervals are pointwise paired item-bootstrap intervals,
conditional on this seed's models and selected checkpoints. No statement about
training-seed uncertainty or a confirmatory decision is made.
"""
    combined = intro + navigation + "\n---\n\n" + "\n---\n\n".join(sections)
    combined += "\n---\n\n## Non-binding confirmatory notes\n\n" + re.sub(r"^(#{1,5}) ", r"\1## ", notes_snapshot, flags=re.M)
    (PACKAGE / "README.md").write_text(combined)
    assert all(row["execution_count"] is not None and not any(o["output_type"] == "error" for o in row["outputs"])
               for row in cells if row["cell_type"] == "code")
    assert len({r["id"] for r in cells}) == len(cells)
    print(f"PASS: {execution_count} code cells executed; notebook and combined Markdown written.")


if __name__ == "__main__":
    main()
