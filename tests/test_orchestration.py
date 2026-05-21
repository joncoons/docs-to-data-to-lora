"""Smoke test the CLI parses args correctly."""
import subprocess
import sys


def test_cli_dry_run(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/build_v2_dataset.py",
         "--collection", "nim_curated",
         "--output", str(tmp_path),
         "--stage", "0",
         "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "DRY RUN" in result.stderr or "DRY RUN" in result.stdout
