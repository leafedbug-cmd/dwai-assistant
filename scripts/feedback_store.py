import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def default_feedback_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "feedback" / "feedback.jsonl"


def append_feedback(record: Dict[str, Any], path: Optional[str] = None) -> Path:
    out_path = Path(path) if path else default_feedback_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = dict(record)
    rec.setdefault("ts", datetime.now(timezone.utc).isoformat())

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path

