import json
import subprocess
import sys
from pathlib import Path

from tools.build_notebook import convert


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "musc_demo_colab_runbook.md"
NOTEBOOK = ROOT / "notebooks" / "MuSc_classroom_demo.ipynb"
VI_RUNBOOK = ROOT / "docs" / "musc_demo_colab_runbook_vi.md"
VI_NOTEBOOK = ROOT / "notebooks" / "MuSc_classroom_demo_vi.ipynb"
PINNED_COMMIT = "b76b93da8bd3096a99964a96ae29d46f197a0651"


def _code_text(notebook: dict[str, object]) -> str:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return "\n".join(
        str(cell["source"])
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )


def test_builder_uses_output_stem_as_colab_notebook_name(tmp_path):
    source = tmp_path / "runbook.md"
    output = tmp_path / "classroom-demo.ipynb"
    source.write_text("# Demo\n\n```python\nprint('ok')\n```\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_notebook.py"), source, output],
        check=True,
    )
    notebook = json.loads(output.read_text(encoding="utf-8"))

    assert notebook["metadata"]["colab"]["name"] == output.name


def test_demo_runbook_produces_markdown_and_executable_cells():
    notebook = convert(RUNBOOK.read_text(encoding="utf-8"), NOTEBOOK.name)
    cell_types = [cell["cell_type"] for cell in notebook["cells"]]

    assert "markdown" in cell_types
    assert cell_types.count("code") >= 7
    assert all(cell["source"].strip() for cell in notebook["cells"])


def test_demo_notebook_contains_commit_verification_before_app_launch():
    notebook = convert(RUNBOOK.read_text(encoding="utf-8"), NOTEBOOK.name)
    code = _code_text(notebook)

    verification = code.index("rev-parse HEAD")
    pinned_assertion = code.index(PINNED_COMMIT)
    launch = code.index("demo.launch(")
    assert verification < launch
    assert pinned_assertion < launch
    assert 'os.environ["MUSC_EAGER_INIT"] = "0"' in code


def test_demo_notebook_does_not_embed_credentials_or_private_drive_paths():
    notebook = convert(RUNBOOK.read_text(encoding="utf-8"), NOTEBOOK.name)
    serialized = json.dumps(notebook)

    assert "ghp_" not in serialized
    assert "github_pat_" not in serialized
    assert "PRIVATE_REPO_TOKEN=" not in serialized
    assert "/Users/" not in serialized
    assert "/home/" not in serialized
    assert "DuyKhanh" not in serialized


def test_vietnamese_setup_runbook_builds_an_executable_notebook():
    notebook = convert(VI_RUNBOOK.read_text(encoding="utf-8"), VI_NOTEBOOK.name)
    markdown = "\n".join(
        str(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = _code_text(notebook)

    assert "Hướng dẫn thiết lập" in markdown
    assert "T4 GPU" in markdown
    assert "Colab Secrets" in markdown
    assert "musc-classroom-prepared-bundle.zip" in code
    assert "ZipFile" in code
    assert "manifest.json" in code
    assert code.index("rev-parse") < code.index("demo.launch(")
    assert 'os.environ["MUSC_EAGER_INIT"] = "0"' in code


def test_vietnamese_setup_notebook_keeps_credentials_out_of_source():
    notebook = convert(VI_RUNBOOK.read_text(encoding="utf-8"), VI_NOTEBOOK.name)
    serialized = json.dumps(notebook)

    assert "ghp_" not in serialized
    assert "github_pat_" not in serialized
    assert "PRIVATE_REPO_TOKEN=" not in serialized
    assert "/Users/" not in serialized
    assert "/home/" not in serialized
