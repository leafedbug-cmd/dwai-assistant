import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_feedback(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def normalize_variant_parts(vp: Any) -> Dict[str, List[str]]:
    if not isinstance(vp, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in vp.items():
        if not k:
            continue
        if isinstance(v, list):
            pns = [str(p).strip() for p in v if str(p).strip()]
        else:
            pns = [str(v).strip()] if str(v).strip() else []
        if pns:
            out[str(k).strip()] = pns
    return out


def to_instruction_example(rec: Dict[str, Any]) -> Dict[str, Any] | None:
    model = (rec.get("model") or "").strip()
    part = (rec.get("part") or "").strip()
    if not model or not part:
        return None

    corrected = rec.get("corrected") or {}
    corrected_vp = normalize_variant_parts((corrected.get("variant_parts") if isinstance(corrected, dict) else None))
    observed_vp = normalize_variant_parts(rec.get("variant_parts"))

    rating = (rec.get("rating") or "").lower()
    vp = corrected_vp or (observed_vp if rating == "correct" else {})
    if not vp:
        return None

    instruction = (
        "You are a parts assistant. Given a machine model and a short part description, "
        "return part numbers grouped by model variant/iteration when applicable."
    )
    inp = f"Model: {model}\nPart: {part}"

    lines = ["Part numbers by variant:"]
    for variant, pns in vp.items():
        lines.append(f"- {variant}: {', '.join(pns)}")
    output = "\n".join(lines).strip()

    return {"instruction": instruction, "input": inp, "output": output}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export feedback into a simple instruction-tuning JSONL dataset.")
    ap.add_argument("--feedback", required=True, help="Path to feedback.jsonl")
    ap.add_argument("--out", required=True, help="Output path for dataset jsonl")
    args = ap.parse_args()

    feedback_path = Path(args.feedback)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_feedback(feedback_path)
    examples = []
    for r in records:
        ex = to_instruction_example(r)
        if ex:
            examples.append(ex)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

