from pathlib import Path
from dtr.cli import demo

def test_demo(tmp_path: Path):
    out = tmp_path / "artifacts"
    demo.callback = None  # Typer appeasement for pytest
    demo(episodes=5, out=str(out))
    assert (out / "summary.json").exists()
