import argparse
import json
import os
import sys
import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional
from functools import lru_cache

import numpy as np
import requests
import hnswlib
import fitz  # PyMuPDF
import base64


DEFAULT_ASSISTANT_NAME = "dwFixIT"
DEFAULT_PERSONA_RULES = """\
You are dwFixIT, a Ditch Witch field service assistant for technicians.

Goals:
- Diagnose machine issues from plain-language symptoms.
- Identify parts from descriptions when the user doesn't know official names.

Voice:
- Concise, practical, no fluff. Use short checklists and clear next steps.

Rules:
- Use the provided CONTEXT as the source of truth. If the answer is not in CONTEXT, say so and ask for the minimum missing info.
- Map user terms to likely official terms/synonyms (e.g., "brain box" -> controller/ECU; "spinny wheel" -> idler/roller/pulley).
- Do NOT ask for serial/model up front for parts requests.
- Only ask for serial number if the manual page indicates a serial-number break/variant note, or if multiple close matches remain.
- If a serial-number break exists, list each variant with its break condition, then ask for the serial to pick the correct one.
- Safety first: warn before energized work, relieving hydraulic pressure, lifting/rotating components; stop and ask if unsafe/unclear.

Output formats:
Parts requests:
1) Likely official part name(s) + synonyms
2) Matches found (options with part numbers/callouts when present)
3) How to confirm (1-2 quick checks)
4) If serial break applies: ask for serial number

Issue diagnosis:
1) Most likely causes (top 3)
2) Quick checks first (lowest effort / highest probability)
3) Deeper tests only if needed (ask for measurable observations)
"""


def load_index(index_dir: Path):
    index_path = index_dir / "index.bin"
    meta_path = index_dir / "meta.jsonl"
    db_path = index_dir / "meta.sqlite"
    cfg_path = index_dir / "config.json"
    if not index_path.exists() or (not meta_path.exists() and not db_path.exists()):
        raise FileNotFoundError("Index not found. Run rag_reindex.py first.")
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Prefer SQLite for fast random access; loading 300k+ chunks into memory is slow and huge.
    if not db_path.exists():
        raise FileNotFoundError(
            f"Missing metadata database: {db_path}. Re-run rag_reindex.py (it should generate meta.sqlite)."
        )

    return index_path, db_path, cfg


def fetch_meta_records(db_path: Path, ids: List[int]) -> List[dict]:
    if not ids:
        return []
    uniq = list(dict.fromkeys(int(i) for i in ids if i is not None))
    if not uniq:
        return []

    placeholders = ",".join("?" for _ in uniq)
    rows: Dict[int, dict] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for rid, path, chunk_index, page_number, text in conn.execute(
            f"SELECT id,path,chunk_index,page_number,text FROM chunks WHERE id IN ({placeholders})",
            tuple(uniq),
        ):
            rows[int(rid)] = {
                "id": int(rid),
                "path": str(path),
                "chunk_index": int(chunk_index),
                "page_number": int(page_number),
                "text": str(text),
            }
    finally:
        conn.close()

    out = []
    for i in ids:
        rec = rows.get(int(i))
        if rec:
            out.append(rec)
    return out


def embed(text: str, base_url: str, model: str, session: requests.Session) -> np.ndarray:
    # Prefer batch embedding endpoint; fall back to legacy endpoint.
    url_embed = base_url.rstrip("/") + "/api/embed"
    resp = session.post(url_embed, json={"model": model, "input": [text]}, timeout=300)
    if resp.status_code == 404:
        url = base_url.rstrip("/") + "/api/embeddings"
        r = session.post(url, json={"model": model, "prompt": text}, timeout=300)
        r.raise_for_status()
        vec = np.array(r.json()["embedding"], dtype=np.float32)
    else:
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data and data["embeddings"]:
            vec = np.array(data["embeddings"][0], dtype=np.float32)
        elif "data" in data and data["data"]:
            vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected /api/embed response shape: {list(data.keys())}")
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec


MODEL_TOKEN_RE = re.compile(r"\bJT\s*(\d{2,4})([A-Z0-9]{0,4})\b", re.IGNORECASE)
PART_NO_RE = re.compile(r"\b\d{3}-\d{3,5}\b")
STRUCT_MODEL_RE = re.compile(r"(?im)^\s*model\s*:\s*(.+?)\s*$")
STRUCT_PART_RE = re.compile(r"(?im)^\s*(part|part description|short part description)\s*:\s*(.+?)\s*$")
STRUCT_Q_RE = re.compile(r"(?im)^\s*(question|service question)\s*:\s*(.+?)\s*$")


def normalize_model_token(token: str) -> str:
    return re.sub(r"\s+", "", (token or "").upper())


def extract_model_tokens(text: str) -> List[str]:
    tokens = []
    for digits, suffix in MODEL_TOKEN_RE.findall(text or ""):
        tok = f"JT{digits}{suffix}".upper()
        tokens.append(tok)
    # de-dupe while preserving order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def is_parts_question(question: str) -> bool:
    q = (question or "").lower()
    triggers = [
        "part number",
        "part #",
        "p/n",
        "pn ",
        "callout",
        "item #",
        "item number",
        "filter",
        "belt",
        "o-ring",
        "oring",
    ]
    return any(t in q for t in triggers)


def parse_structured_fields(question: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    q = question or ""
    model = None
    part = None
    service_q = None

    m = STRUCT_MODEL_RE.search(q)
    if m:
        model = m.group(1).strip() or None
    for m in STRUCT_PART_RE.finditer(q):
        part = (m.group(2) or "").strip() or part
    for m in STRUCT_Q_RE.finditer(q):
        service_q = (m.group(2) or "").strip() or service_q

    return model, part, service_q


def variant_from_path(path: str) -> str:
    """
    Try to label the model iteration/variant from the PDF path.
    Examples:
      directional_drills/.../JT2020 MACH 1 TIER 3/... -> JT2020 MACH 1 TIER 3
      directional_drills/LEGACY/JT2020 MACH 1/...     -> JT2020 MACH 1 (LEGACY)
    """
    p = (path or "").replace("\\", "/")
    parts = [seg for seg in p.split("/") if seg]
    variant_seg = None
    for seg in parts:
        if MODEL_TOKEN_RE.search(seg):
            variant_seg = seg.strip()
            break
    if not variant_seg:
        return "Unknown variant"
    if "LEGACY" in (s.upper() for s in parts):
        if "LEGACY" not in variant_seg.upper():
            return f"{variant_seg} (LEGACY)"
    return variant_seg


def extract_part_numbers(text: str) -> List[str]:
    seen = set()
    out = []
    for m in PART_NO_RE.finditer(text or ""):
        pn = m.group(0)
        if pn not in seen:
            seen.add(pn)
            out.append(pn)
    return out


def rerank_hits_for_models(
    hits: List[Tuple[dict, float]],
    question_models: List[str],
) -> List[Tuple[dict, float]]:
    """
    If the user provided a 4-digit model token (e.g., JT2020/JT2720), heavily prefer hits
    whose path/text contains that token to avoid near-collisions like JT7020.
    """
    if not hits or not question_models:
        return hits

    q_models_norm = [normalize_model_token(t) for t in question_models]
    strong = [t for t in q_models_norm if re.match(r"^JT\d{4}[A-Z0-9]{0,4}$", t)]
    if not strong:
        return hits

    def contains_any_strong(rec: dict) -> bool:
        p = normalize_model_token(rec.get("path", ""))
        txt = normalize_model_token(rec.get("text", ""))
        return any(s in p or s in txt for s in strong)

    strong_hits = [(rec, dist) for rec, dist in hits if contains_any_strong(rec)]
    if strong_hits:
        return strong_hits
    return hits


def default_feedback_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "feedback" / "feedback.jsonl"


@lru_cache(maxsize=1)
def load_feedback_index(feedback_path: str) -> Dict[Tuple[str, str], Dict[str, List[str]]]:
    """
    Returns a mapping: (model_base, part_key) -> {variant_label: [part_numbers...]}
    Only includes records marked correct or with corrected part numbers.
    """
    path = Path(feedback_path)
    if not path.exists():
        return {}

    index: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            model = normalize_model_token(rec.get("model") or "")
            part = (rec.get("part") or "").strip().lower()
            if not model or not part:
                continue

            rating = (rec.get("rating") or "").lower()
            corrected = rec.get("corrected") or {}
            corrected_parts = corrected.get("variant_parts") if isinstance(corrected, dict) else None
            observed_parts = rec.get("variant_parts")

            variant_parts: Optional[Dict[str, List[str]]] = None
            if isinstance(corrected_parts, dict) and corrected_parts:
                variant_parts = {str(k): [str(p) for p in (v or [])] for k, v in corrected_parts.items()}
            elif rating == "correct" and isinstance(observed_parts, dict) and observed_parts:
                variant_parts = {str(k): [str(p) for p in (v or [])] for k, v in observed_parts.items()}

            if not variant_parts:
                continue

            model_base = model[:6] if re.match(r"^JT\d{4}", model) else model
            key = (model_base, part)
            bucket = index.setdefault(key, {})
            for variant, pns in variant_parts.items():
                if not pns:
                    continue
                vb = bucket.setdefault(variant, [])
                for pn in pns:
                    if pn not in vb:
                        vb.append(pn)

    return index


def format_variant_parts_answer(model: str, part: str, variant_parts: Dict[str, List[str]]) -> str:
    lines = []
    lines.append(f"Model: {model}")
    lines.append(f"Part: {part}")
    lines.append("")
    lines.append("Part numbers by model variant:")
    for variant, pns in variant_parts.items():
        if not pns:
            continue
        lines.append(f"- {variant}: {', '.join(pns)}")
    lines.append("")
    lines.append("If you can provide the serial number, I can pick the exact variant when a serial break applies.")
    return "\n".join(lines).strip()


def generate(prompt: str, base_url: str, model: str, session: requests.Session) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    resp = session.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def chat_vision(prompt: str, images_b64: List[str], base_url: str, model: str, session: requests.Session) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    msg = {"role": "user", "content": prompt, "images": images_b64}
    resp = session.post(url, json={"model": model, "messages": [msg], "stream": False}, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


def render_pdf_pages(pdf_path: Path, page_numbers_1_based: List[int]) -> List[str]:
    images = []
    doc = fitz.open(pdf_path)
    try:
        for pn in page_numbers_1_based:
            if pn < 1 or pn > doc.page_count:
                continue
            page = doc.load_page(pn - 1)
            pix = page.get_pixmap(dpi=200, alpha=False)
            png_bytes = pix.tobytes("png")
            images.append(base64.b64encode(png_bytes).decode("utf-8"))
    finally:
        doc.close()
    return images


def should_use_vision(question: str, hits: List[tuple]) -> bool:
    q = question.lower()
    keyword_triggers = [
        "diagram",
        "exploded",
        "callout",
        "call out",
        "figure",
        "item #",
        "item number",
        "reference number",
        "parts list",
        "parts manual",
        "label",
        "shown",
        "see figure",
    ]
    if any(k in q for k in keyword_triggers):
        return True
    if not hits:
        return False
    best_dist = hits[0][1]
    if best_dist > 0.35:
        return True
    top_text = hits[0][0].get("text", "") or ""
    if len(top_text) < 200:
        return True
    non_space = sum(1 for ch in top_text if not ch.isspace())
    if non_space == 0:
        return True
    digits = sum(1 for ch in top_text if ch.isdigit())
    letters = sum(1 for ch in top_text if ch.isalpha())
    if digits / non_space > 0.2 and letters / non_space < 0.5:
        return True
    return False


def build_prompt(question: str, context: str, assistant_name: str, persona_rules: str) -> str:
    rules = persona_rules.strip() or DEFAULT_PERSONA_RULES.strip()
    name = assistant_name.strip() or DEFAULT_ASSISTANT_NAME
    return (
        f"{rules}\n\n"
        f"Identity:\n- Name: {name}\n\n"
        "Instructions:\n"
        "- Answer using the CONTEXT from service documents.\n"
        "- Cite sources like [1], [2] matching the context blocks.\n\n"
        "- If multiple model variants appear in CONTEXT (e.g., Tier versions / Mach 1 / Legacy), group part numbers by variant and state which variants each part fits.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "ANSWER:"
    )


def build_vision_prompt(question: str, assistant_name: str, persona_rules: str) -> str:
    rules = persona_rules.strip() or DEFAULT_PERSONA_RULES.strip()
    name = assistant_name.strip() or DEFAULT_ASSISTANT_NAME
    return (
        f"{rules}\n\n"
        f"Identity:\n- Name: {name}\n\n"
        "The images are pages from a parts/service manual.\n"
        "Use diagrams, callout numbers, and parts lists to answer.\n"
        "If possible, map callout numbers to part names and part numbers.\n\n"
        f"QUESTION:\n{question}\n"
    )


def answer_question(
    question: str,
    config_path: str | None = None,
    index_dir_override: str | None = None,
    force_vision: bool = False,
    disable_vision: bool = False,
) -> dict:
    q = (question or "").strip()
    if not q:
        raise ValueError("Provide a question.")

    structured_model, structured_part, _structured_service_q = parse_structured_fields(q)
    question_models = extract_model_tokens(q)
    parts_mode = is_parts_question(q)

    cfg = {}
    if config_path:
        cp = Path(config_path)
        if not cp.exists():
            raise FileNotFoundError(
                f"Config file not found: {cp}. Create it with Copy-Item .\\scripts\\rag_config.example.json .\\scripts\\rag_config.json"
            )
        cfg = json.loads(cp.read_text(encoding="utf-8"))

    docs_root = Path(cfg.get("docs_root", Path("docs/service-documents").resolve()))
    index_dir = Path(index_dir_override or cfg.get("index_dir", Path("data/rag").resolve()))
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    embed_model = cfg.get("embedding_model", "nomic-embed-text")
    chat_model = cfg.get("chat_model", "qwen3:8b")
    vision_model = cfg.get("vision_model", "qwen2.5-vl:7b")
    vision_max_pages = int(cfg.get("vision_max_pages", 3))
    top_k = int(cfg.get("top_k", 4))
    max_context_chars = int(cfg.get("max_context_chars", 12000))
    assistant_name = cfg.get("assistant_name", DEFAULT_ASSISTANT_NAME)
    persona_rules = cfg.get("persona_rules", DEFAULT_PERSONA_RULES)
    feedback_path = cfg.get("feedback_path") or str(default_feedback_path())

    if structured_model and structured_model not in question_models:
        question_models = extract_model_tokens(structured_model) + question_models

    # Fast path: if we have confirmed part numbers from feedback, answer without an LLM call.
    if parts_mode and structured_model and structured_part:
        fb = load_feedback_index(str(feedback_path))
        model_norm = normalize_model_token(structured_model)
        model_base = model_norm[:6] if re.match(r"^JT\\d{4}", model_norm) else model_norm
        key = (model_base, structured_part.strip().lower())
        known = fb.get(key) or {}
        if known:
            return {
                "answer": format_variant_parts_answer(structured_model, structured_part, known),
                "sources": [],
                "used_vision": False,
                "hits": [],
                "variant_parts": known,
                "feedback_used": True,
            }

    index_path, db_path, stored_cfg = load_index(index_dir)
    if stored_cfg:
        base_url = stored_cfg.get("ollama_base_url", base_url)
        embed_model = stored_cfg.get("embedding_model", embed_model)

    session = requests.Session()

    # Nudge embedding toward the right corpus when the user includes an explicit model token.
    # This helps keep JT2020 from drifting into JT7020, etc.
    embed_query = q
    if question_models:
        embed_query = f"{q}\n\nModel: {' '.join(question_models)}"
        if parts_mode:
            embed_query += "\nDocument type: parts manual"

    q_vec = embed(embed_query, base_url, embed_model, session)

    dim = q_vec.shape[0]
    idx = get_hnsw_index(str(index_path), dim)

    # Pull more candidates so we can filter/rerank, then take a diverse top_k.
    candidate_k = max(top_k * 10, 40)
    labels, distances = idx.knn_query(q_vec, k=candidate_k)
    ids = [int(l) for l in labels[0].tolist()]
    recs = fetch_meta_records(db_path, ids)
    hits: List[Tuple[dict, float]] = []
    for rec, dist in zip(recs, distances[0].tolist()):
        if rec:
            hits.append((rec, float(dist)))

    hits = rerank_hits_for_models(hits, question_models)

    # Prefer "Complete Parts Book" / "Parts" content for parts queries.
    if parts_mode:
        def parts_boost(rec: dict, dist: float) -> float:
            p = (rec.get("path", "") or "").lower()
            bonus = 0.0
            if "complete parts book" in p or "parts book" in p:
                bonus += 0.04
            if "parts manual" in p:
                bonus += 0.03
            if "filters" in p:
                bonus += 0.02
            return dist - bonus

        hits.sort(key=lambda t: parts_boost(t[0], t[1]))
    else:
        hits.sort(key=lambda t: t[1])

    # Diversify across model variants (so we surface JT2020 Tier 3 vs JT2020 Legacy/M1, etc).
    selected: List[Tuple[dict, float]] = []
    seen_variants = set()
    for rec, dist in hits:
        variant = variant_from_path(rec.get("path", ""))
        if variant not in seen_variants:
            selected.append((rec, dist))
            seen_variants.add(variant)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        for rec, dist in hits:
            if (rec, dist) in selected:
                continue
            selected.append((rec, dist))
            if len(selected) >= top_k:
                break

    hits = selected

    context_blocks = []
    used = 0
    citations = []
    for rec, _dist in hits:
        block = f"[{len(citations)+1}] {rec['path']}\n{rec['text']}\n"
        if used + len(block) > max_context_chars:
            break
        context_blocks.append(block)
        used += len(block)
        citations.append(rec["path"])

    context = "\n".join(context_blocks)
    prompt = build_prompt(q, context, assistant_name, persona_rules)
    answer = generate(prompt, base_url, chat_model, session)

    variant_parts: Dict[str, List[str]] = {}
    if parts_mode and hits:
        for rec, _dist in hits:
            variant = variant_from_path(rec.get("path", ""))
            pns = extract_part_numbers(rec.get("text", ""))
            if not pns:
                continue
            bucket = variant_parts.setdefault(variant, [])
            for pn in pns:
                if pn not in bucket:
                    bucket.append(pn)

    used_vision = False
    use_vision = force_vision or (not disable_vision and should_use_vision(q, hits))
    if use_vision and hits:
        pages = []
        seen = set()
        for rec, _dist in hits:
            pn = rec.get("page_number")
            if pn and pn not in seen:
                seen.add(pn)
                pages.append(int(pn))
            if len(pages) >= vision_max_pages:
                break

        if pages:
            pdf_path = docs_root / hits[0][0]["path"]
            if pdf_path.exists():
                images_b64 = render_pdf_pages(pdf_path, pages)
                if images_b64:
                    vision_prompt = build_vision_prompt(q, assistant_name, persona_rules)
                    try:
                        vision_answer = chat_vision(vision_prompt, images_b64, base_url, vision_model, session)
                        if vision_answer:
                            answer = vision_answer
                            used_vision = True
                    except Exception:
                        pass

    return {
        "answer": answer,
        "sources": citations,
        "used_vision": used_vision,
        "hits": [{"path": rec.get("path"), "page_number": rec.get("page_number"), "dist": dist} for rec, dist in hits],
        "variant_parts": variant_parts,
        "feedback_used": False,
    }


_HNSW_CACHE: Dict[Tuple[str, int], hnswlib.Index] = {}


def get_hnsw_index(index_path: str, dim: int) -> hnswlib.Index:
    key = (index_path, int(dim))
    idx = _HNSW_CACHE.get(key)
    if idx is None:
        idx = hnswlib.Index(space="cosine", dim=int(dim))
        idx.load_index(index_path)
        idx.set_ef(50)
        _HNSW_CACHE[key] = idx
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask questions over the local RAG index.")
    ap.add_argument("question", nargs="*", help="Question to ask.")
    ap.add_argument("--config", help="Path to rag_config.json (optional).")
    ap.add_argument("--index-dir", help="Override index directory.")
    ap.add_argument("--vision", action="store_true", help="Force vision mode on.")
    ap.add_argument("--no-vision", action="store_true", help="Disable auto vision.")
    args = ap.parse_args()
    question = " ".join(args.question).strip()
    if not question:
        print("Provide a question.", file=sys.stderr)
        return 2

    try:
        result = answer_question(
            question=question,
            config_path=args.config,
            index_dir_override=args.index_dir,
            force_vision=args.vision,
            disable_vision=args.no_vision,
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    print(result["answer"])
    if result.get("sources"):
        print("\nSources:")
        for i, p in enumerate(result["sources"], start=1):
            print(f"[{i}] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
