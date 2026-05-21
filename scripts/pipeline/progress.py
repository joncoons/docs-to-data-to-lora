"""Resume-aware progress checkpoint."""
import json
from pathlib import Path


class Progress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = {"completed_stages": []}
        if path.exists():
            with path.open() as f:
                self.data = json.load(f)

    def is_done(self, stage: str) -> bool:
        return stage in self.data.get("completed_stages", [])

    def mark_done(self, stage: str) -> None:
        if stage not in self.data["completed_stages"]:
            self.data["completed_stages"].append(stage)
        with self.path.open("w") as f:
            json.dump(self.data, f, indent=2)
