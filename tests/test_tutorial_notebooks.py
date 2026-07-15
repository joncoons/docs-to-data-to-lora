from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "tutorial" / "notebooks"


def iter_notebooks() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def test_tutorial_notebooks_are_stored_without_outputs() -> None:
    for notebook_path in iter_notebooks():
        notebook = json.loads(notebook_path.read_text())
        for idx, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, (
                f"{notebook_path.name} cell {idx} has execution_count"
            )
            assert cell.get("outputs", []) == [], (
                f"{notebook_path.name} cell {idx} has stored outputs"
            )


def test_tutorial_notebooks_execute_in_dry_run_mode(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    for notebook_path in iter_notebooks():
        namespace = {"__name__": "__main__"}
        notebook = json.loads(notebook_path.read_text())
        for idx, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            code = compile(source, f"{notebook_path}:cell{idx}", "exec")
            exec(code, namespace)
