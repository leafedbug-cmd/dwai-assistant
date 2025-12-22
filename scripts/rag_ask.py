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

    rows: Dict[int, dict] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        # Chunk large IN(...) lists to avoid SQLite variable limits on some builds.
        chunk_size = 900
        for off in range(0, len(uniq), chunk_size):
            chunk = uniq[off : off + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            for rid, path, chunk_index, page_number, text in conn.execute(
                f"SELECT id,path,chunk_index,page_number,text FROM chunks WHERE id IN ({placeholders})",
                tuple(chunk),
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


# Extract common Ditch Witch model tokens from free text and PDF paths.
# Keep the prefix whitelist tight to avoid matching unrelated IDs (e.g., "ID043...").
MODEL_TOKEN_RE = re.compile(
    r"\b(JT|FXT|FX|HXT|HX|MV|FT|SK|RT|ST|PT)\s*(\d{1,4})([A-Z0-9]{0,6})\b",
    re.IGNORECASE,
)
PART_NO_RE = re.compile(r"\b\d{3}-\d{3,5}\b")
STRUCT_MODEL_RE = re.compile(r"(?im)^\s*model\s*:\s*(.+?)\s*$")
STRUCT_PART_RE = re.compile(r"(?im)^\s*(part|part description|short part description)\s*:\s*(.+?)\s*$")
STRUCT_Q_RE = re.compile(r"(?im)^\s*(question|service question)\s*:\s*(.+?)\s*$")
RE_FILTER_ENTRY_PN_FIRST = re.compile(
    r"(?P<pn>\d{3}-\d{3,5})\s+(?P<desc>.{0,140}?\bfilter\b.{0,140}?)(?=\s+\d{3}-\d{3,5}\b|$)",
    re.IGNORECASE,
)
RE_FILTER_ENTRY_DESC_FIRST = re.compile(
    r"(?P<desc>.{0,140}?\bfilter\b.{0,140}?)\s+(?P<pn>\d{3}-\d{3,5})",
    re.IGNORECASE,
)


def normalize_model_token(token: str) -> str:
    return re.sub(r"\s+", "", (token or "").upper())


def normalize_part_key(part: str) -> str:
    s = (part or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_multi_part_key(part: str) -> bool:
    s = normalize_part_key(part)
    # Common separators that imply the user asked for multiple unrelated parts.
    return bool(re.search(r"\b(and|or)\b|[,&/+]|\bplus\b", s))


def canonicalize_part_key(part: str, question_text: str | None = None) -> str:
    """
    Canonicalize common "non-textbook" phrasing so feedback matches more often.

    Conservative by design: only rewrites when the intent is clear.
    """
    p = normalize_part_key(part)
    q = normalize_part_key(question_text or "")
    combined = f"{p} {q}".strip()

    # Filters
    if "filter" in combined:
        if "hydraulic" in combined:
            return "hydraulic oil filter"
        if "fuel" in combined or "water separator" in combined or "separator" in combined:
            return "fuel filter"
        if "air" in combined or "intake" in combined:
            return "air filter"
        if "engine oil" in combined:
            return "engine oil filter"
        if "oil" in combined and "filter" in combined and "hydraulic" not in combined:
            return "engine oil filter"

    # Ignition switch ("key switch", "starter switch", etc.)
    if any(t in combined for t in ("ignition", "key switch", "keyswitch", "start switch", "starter switch")) and "switch" in combined:
        return "ignition switch"

    return p


def part_key_category(part: str) -> str:
    p = normalize_part_key(part)
    if "filter" in p or "element" in p or "separator" in p:
        return "filter"
    if "switch" in p:
        return "switch"
    if "belt" in p:
        return "belt"
    if "o ring" in p or "oring" in p:
        return "oring"
    return "other"


def part_key_tokens(part: str) -> List[str]:
    p = normalize_part_key(part)
    toks = [t for t in p.split() if t]
    stop = {"the", "a", "an", "for", "of", "and", "or", "to", "on", "in", "with", "kit", "assy", "assembly"}
    toks = [t for t in toks if t not in stop]
    return toks


def normalize_model_match_key(s: str) -> str:
    """
    Normalization used for model matching in paths/text:
    - uppercase
    - drop all non A-Z/0-9
    This makes matching robust to punctuation/spaces like "JT10," or "JT10/".
    """
    return re.sub(r"[^A-Z0-9]+", "", (s or "").upper())


def extract_model_tokens(text: str) -> List[str]:
    tokens = []
    for prefix, digits, suffix in MODEL_TOKEN_RE.findall(text or ""):
        tok = f"{prefix}{digits}{suffix}".upper()
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
    If the user provided model tokens (e.g., JT10, JT2020), prefer hits whose PDF path
    contains an exact model segment. This avoids cross-model drift (e.g., JT10 -> JT520)
    and near-collisions like JT2020 vs JT7020.
    """
    if not hits or not question_models:
        return hits

    def _expand_aliases(tok: str) -> List[str]:
        t = normalize_model_match_key(tok)
        m = re.match(r"^([A-Z]{2,4})(\d{2,4})([A-Z0-9]{0,6})$", t)
        if not m:
            return [t] if t else []
        prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        out = [t]
        # Common "missing letter" user inputs: FX30 vs FXT30, HX75 vs HXT75.
        if prefix == "FX" and not suffix:
            out.append(f"FXT{digits}")
        if prefix == "HX" and not suffix:
            out.append(f"HXT{digits}")
        return list(dict.fromkeys(out))

    q_models_norm: List[str] = []
    for t in question_models:
        q_models_norm.extend(_expand_aliases(t))

    strong_any = [
        t
        for t in q_models_norm
        if re.match(r"^[A-Z]{2,4}\d{1,4}[A-Z0-9]{0,6}$", t)
    ]
    if not strong_any:
        return hits

    # Best signal: an exact path segment match (e.g., ".../JT10/..." or ".../JT2020 MACH 1/...").
    def has_exact_model_segment(rec: dict) -> bool:
        p = (rec.get("path", "") or "").replace("\\", "/")
        segs = [normalize_model_match_key(seg) for seg in p.split("/") if seg]

        def _token_matches_seg(tok: str, seg: str) -> bool:
            # For 4-digit JT models, allow variant segments like "JT2020 MACH 1 TIER 3".
            if re.match(r"^JT\d{4}", tok):
                return seg.startswith(tok)
            # For other model families, prefer exact (FX30 should not match FXT30 unless we
            # explicitly aliased it in _expand_aliases()).
            return seg == tok

        return any(_token_matches_seg(tok, seg) for tok in strong_any for seg in segs)

    segment_hits = [(rec, dist) for rec, dist in hits if has_exact_model_segment(rec)]
    if segment_hits:
        return segment_hits

    def contains_any_strong(rec: dict) -> bool:
        p = normalize_model_match_key(rec.get("path", ""))
        txt = normalize_model_match_key(rec.get("text", ""))
        # For safety, apply substring fallback only for 4-digit JT tokens where collisions are common.
        strong_4 = [s for s in strong_any if re.match(r"^JT\d{4}[A-Z0-9]{0,6}$", s)]
        return any(s in p or s in txt for s in strong_4)

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
            part_raw = normalize_part_key(rec.get("part") or "")
            if not model or not part_raw:
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

            # Always index the raw key.
            keys = [part_raw]
            # Also index a canonical alias (only when it's a single-part request).
            if part_raw and not is_multi_part_key(part_raw):
                canon = canonicalize_part_key(part_raw, rec.get("question") or "")
                if canon and canon != part_raw:
                    keys.append(canon)

            for part_key in keys:
                key = (model_base, part_key)
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


def _parse_timeout_s(cfg: dict, key: str, default: int) -> int:
    v = cfg.get(key, default)
    try:
        return max(1, int(v))
    except Exception:
        return default


def embed(text: str, base_url: str, model: str, session: requests.Session, *, timeout_s: int = 300) -> np.ndarray:
    # Prefer batch embedding endpoint; fall back to legacy endpoint.
    url_embed = base_url.rstrip("/") + "/api/embed"
    resp = session.post(url_embed, json={"model": model, "input": [text]}, timeout=(10, timeout_s))
    if resp.status_code == 404:
        url = base_url.rstrip("/") + "/api/embeddings"
        r = session.post(url, json={"model": model, "prompt": text}, timeout=(10, timeout_s))
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


def generate(prompt: str, base_url: str, model: str, session: requests.Session, *, timeout_s: int = 900) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    # Stream tokens back to avoid client-side read timeouts during long generations
    # (model load, CPU inference, etc.).
    resp = session.post(
        url,
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=(10, timeout_s),
    )
    resp.raise_for_status()

    parts: List[str] = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        chunk = data.get("response")
        if chunk:
            parts.append(str(chunk))
        if data.get("done"):
            break
    return "".join(parts).strip()


def chat_vision(
    prompt: str, images_b64: List[str], base_url: str, model: str, session: requests.Session, *, timeout_s: int = 900
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    msg = {"role": "user", "content": prompt, "images": images_b64}
    resp = session.post(url, json={"model": model, "messages": [msg], "stream": False}, timeout=(10, timeout_s))
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
        "- Never fabricate part numbers, callouts, or procedures. If a specific part number is not present in CONTEXT, say you can't find it in the manuals provided.\n"
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

    def infer_part_label(question: str) -> str:
        ql = (question or "").lower()
        if "engine oil" in ql and "filter" in ql:
            return "engine oil filter"
        if "hydraulic" in ql and "filter" in ql:
            return "hydraulic oil filter"
        if "fuel" in ql and "filter" in ql:
            return "fuel filter"
        if "air" in ql and "filter" in ql:
            return "air filter"
        if "ignition" in ql and "switch" in ql:
            return "ignition switch"
        if "filter" in ql:
            return "filter"
        return "part"

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
    embed_timeout_s = _parse_timeout_s(cfg, "ollama_embed_timeout_s", 300)
    generate_timeout_s = _parse_timeout_s(cfg, "ollama_generate_timeout_s", 900)
    vision_timeout_s = _parse_timeout_s(cfg, "ollama_vision_timeout_s", 900)

    if structured_model:
        # Ensure the explicit Model: field participates in retrieval even when it isn't JT####.
        question_models = extract_model_tokens(structured_model) + question_models

    # Fast path: if we have confirmed part numbers from feedback, answer without an LLM call.
    # This uses conservative canonical + fuzzy matching so users don't need textbook part names.
    if parts_mode and structured_model:
        fb = load_feedback_index(str(feedback_path))
        model_norm = normalize_model_token(structured_model)
        model_base = model_norm[:6] if re.match(r"^JT\d{4}", model_norm) else model_norm

        # Prefer explicit Short part description, but fall back to an inferred label when missing.
        part_for_feedback = structured_part or infer_part_label(q)
        part_norm = normalize_part_key(part_for_feedback)
        part_canon = canonicalize_part_key(part_norm, q)

        candidates: List[str] = []
        for c in (part_norm, part_canon):
            c = normalize_part_key(c)
            if c and c not in candidates:
                candidates.append(c)

        def _is_too_generic(pk: str) -> bool:
            if pk in ("part", "filter", "element"):
                return True
            toks = part_key_tokens(pk)
            return len(toks) < 2

        for pk in candidates:
            if _is_too_generic(pk):
                continue
            known = fb.get((model_base, pk)) or {}
            if known:
                return {
                    "answer": format_variant_parts_answer(structured_model, pk, known),
                    "sources": [],
                    "used_vision": False,
                    "hits": [],
                    "variant_parts": known,
                    "feedback_used": True,
                }

        # Conservative fuzzy match within this model only (avoid cross-part pollution).
        query_key = next((c for c in candidates if not _is_too_generic(c)), "")
        if query_key:
            qcat = part_key_category(query_key)
            qtoks = set(part_key_tokens(query_key))
            if len(qtoks) >= 2:
                best_key = None
                best_score = 0.0
                for (m, pk), vp in fb.items():
                    if m != model_base:
                        continue
                    if not pk or is_multi_part_key(pk):
                        continue
                    if part_key_category(pk) != qcat:
                        continue
                    ktoks = set(part_key_tokens(pk))
                    if not ktoks:
                        continue
                    overlap = len(qtoks & ktoks)
                    if overlap < 2:
                        continue
                    score = overlap / float(min(len(qtoks), len(ktoks)))
                    if score > best_score:
                        best_score = score
                        best_key = pk

                if best_key and best_score >= 0.8:
                    known = fb.get((model_base, best_key)) or {}
                    if known:
                        return {
                            "answer": format_variant_parts_answer(structured_model, best_key, known),
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
        ql = q.lower()
        part_hint = (structured_part or "").strip().lower()
        if not part_hint:
            if "engine oil" in ql and "filter" in ql:
                part_hint = "engine oil filter"
            elif "hydraulic" in ql and "filter" in ql:
                part_hint = "hydraulic oil filter"
            elif "fuel" in ql and "filter" in ql:
                part_hint = "fuel filter"
            elif "air" in ql and "filter" in ql:
                part_hint = "air filter"
        if part_hint:
            embed_query += f"\nTarget part: {part_hint}"
            if "engine oil" in part_hint and "filter" in part_hint:
                embed_query += "\nSynonyms: oil filter, oil filter element, spin-on oil filter"
        embed_query += "\nDocument type: parts manual"

    q_vec = embed(embed_query, base_url, embed_model, session, timeout_s=embed_timeout_s)

    dim = q_vec.shape[0]
    idx = get_hnsw_index(str(index_path), dim)

    # Pull more candidates so we can filter/rerank, then take a diverse top_k.
    candidate_k = max(top_k * 10, 40)
    if question_models:
        # Short model tokens (JT10) are weak semantic signals; retrieve more candidates
        # so we can filter to the correct model's folder.
        candidate_k = max(candidate_k, 400)
        if parts_mode:
            candidate_k = max(candidate_k, 800)

    def _knn_with_k(k: int) -> List[Tuple[dict, float]]:
        # Increase ef to improve recall when asking for many neighbors.
        try:
            idx.set_ef(max(50, int(k)))
        except Exception:
            pass
        labels, distances = idx.knn_query(q_vec, k=int(k))
        ids = [int(l) for l in labels[0].tolist()]
        recs = fetch_meta_records(db_path, ids)
        out: List[Tuple[dict, float]] = []
        for rec, dist in zip(recs, distances[0].tolist()):
            if rec:
                out.append((rec, float(dist)))
        return out

    hits = _knn_with_k(candidate_k)
    filtered = rerank_hits_for_models(hits, question_models)

    def _looks_like_parts_pdf(rec: dict) -> bool:
        p = (rec.get("path", "") or "").lower()
        return any(k in p for k in ("complete parts book", "parts manual", "parts book", "suggested parts"))

    def _expand_model_aliases(tok: str) -> List[str]:
        t = normalize_model_match_key(tok)
        m = re.match(r"^([A-Z]{2,4})(\d{2,4})([A-Z0-9]{0,6})$", t)
        if not m:
            return [t] if t else []
        prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        out = [t]
        if prefix == "FX" and not suffix:
            out.append(f"FXT{digits}")
        if prefix == "HX" and not suffix:
            out.append(f"HXT{digits}")
        return list(dict.fromkeys(out))

    def _model_match_exists(hits: List[Tuple[dict, float]], model_tokens: List[str]) -> bool:
        toks: List[str] = []
        for t in model_tokens:
            toks.extend(_expand_model_aliases(t))
        toks = [t for t in toks if t]
        if not toks:
            return False

        for rec, _d in hits:
            p = (rec.get("path", "") or "").replace("\\", "/")
            segs = [normalize_model_match_key(seg) for seg in p.split("/") if seg]
            for tok in toks:
                for seg in segs:
                    if tok.startswith("JT") and re.match(r"^JT\d{4}", tok):
                        if seg.startswith(tok):
                            return True
                    else:
                        if seg == tok:
                            return True
        return False

    if question_models:
        # If we still have no model-specific hits, try a larger retrieval to find the
        # correct model folder, then filter again.
        if len(filtered) == len(hits):
            hits2 = _knn_with_k(max(candidate_k, 2000))
            filtered2 = rerank_hits_for_models(hits2, question_models)
            hits = filtered2 if len(filtered2) < len(hits2) else hits2
        else:
            hits = filtered

        # For parts queries, ensure we retrieve at least one likely parts PDF (operator manuals
        # often mention "filter" but don't include part numbers).
        if parts_mode and hits and not any(_looks_like_parts_pdf(rec) for rec, _d in hits):
            hits2 = _knn_with_k(max(candidate_k, 2000))
            filtered2 = rerank_hits_for_models(hits2, question_models)
            if filtered2 and any(_looks_like_parts_pdf(rec) for rec, _d in filtered2):
                hits = filtered2
    else:
        hits = filtered

    # If the user explicitly provided a model but we can't find any matching paths,
    # don't drift into unrelated models (especially for parts lookup).
    if parts_mode and structured_model and question_models and not _model_match_exists(hits, question_models):
        model_norm = normalize_model_match_key(structured_model)
        part_label = (structured_part or "").strip() or "part"
        return {
            "answer": (
                f"Model: {structured_model}\n"
                f"Part: {part_label}\n\n"
                f"I couldn't find any manuals in your index that match model `{model_norm}`.\n"
                "Double-check the model (e.g. `FXT30` vs `FX30`) or confirm the exact machine designation shown on the decal."
            ).strip(),
            "sources": [],
            "used_vision": False,
            "hits": [],
            "variant_parts": {},
            "feedback_used": False,
        }

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

    ranked_hits = hits

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

    def guess_part_keywords(question: str, structured_part: str | None) -> List[str]:
        keywords: List[str] = []
        if structured_part and structured_part.strip():
            keywords.append(structured_part.strip().lower())
        ql = (question or "").lower()
        if "engine oil" in ql and "filter" in ql:
            keywords += ["engine oil filter", "oil filter element", "oil filter"]
        if "hydraulic" in ql and "filter" in ql:
            keywords += ["hydraulic oil filter", "hydraulic filter", "filter element"]
        if "fuel" in ql and "filter" in ql:
            keywords += ["fuel filter", "fuel filter element", "filter element"]
        if "air" in ql and "filter" in ql:
            keywords += ["air filter", "air filter element"]
        if "oil filter" in ql:
            keywords += ["oil filter", "oil filter element"]
        if "filter" in ql:
            keywords += ["filter"]
        # de-dupe while preserving order
        seen = set()
        out = []
        for k in keywords:
            k = (k or "").strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(k)

        # If we have any specific keywords, drop generic ones that cause false positives.
        if any(k not in ("filter", "filter element") for k in out):
            out = [k for k in out if k not in ("filter", "filter element")]
        return out

    def extract_part_numbers_near_keywords(
        text: str, keywords: List[str], *, window: int = 140, max_pns: int = 3
    ) -> List[str]:
        if not text or not keywords:
            return []
        tl = text.lower()

        # Prefer the earliest keyword that yields part numbers, and prefer the part number(s)
        # closest to the keyword occurrence to avoid grabbing adjacent table rows.
        for kw in keywords:
            candidates_after: Dict[str, int] = {}  # pn -> best distance (prefer)
            candidates_before: Dict[str, int] = {}  # pn -> best distance
            start = 0
            while True:
                i = tl.find(kw, start)
                if i < 0:
                    break
                lo = max(0, i - window)
                hi = min(len(text), i + len(kw) + window)
                snippet = text[lo:hi]
                kw_start = i - lo
                kw_end = kw_start + len(kw)
                for m in PART_NO_RE.finditer(snippet):
                    pn = m.group(0)
                    pos = m.start()
                    dist = abs(pos - kw_start)
                    if pos >= kw_end:
                        prev = candidates_after.get(pn)
                        if prev is None or dist < prev:
                            candidates_after[pn] = dist
                    elif pos < kw_start:
                        prev = candidates_before.get(pn)
                        if prev is None or dist < prev:
                            candidates_before[pn] = dist
                start = i + max(1, len(kw))

            best_after = min(candidates_after.values()) if candidates_after else None
            best_before = min(candidates_before.values()) if candidates_before else None

            if best_after is None and best_before is None:
                continue

            # Choose the side whose closest part number is actually closest to the keyword.
            # This handles both formats:
            # - "<desc> <pn>" (after)
            # - "<pn> <desc>" (before)
            use_after = best_after is not None and (best_before is None or best_after <= best_before)
            candidates = candidates_after if use_after else candidates_before

            if candidates:
                ranked = sorted(candidates.items(), key=lambda t: t[1])
                best_dist = ranked[0][1]
                threshold = max(20, best_dist + 10)
                close = [(pn, d) for pn, d in ranked if d <= threshold]
                chosen = close if close else ranked
                return [pn for pn, _d in chosen[: max(1, int(max_pns))]]

        return []

    def format_parts_from_hits(
        model: str,
        part: str,
        hits: List[Tuple[dict, float]],
        keywords: List[str],
    ) -> tuple[str, Dict[str, List[str]], List[str], List[dict]]:
        # Map (variant, pn) -> citation index.
        pn_cite: Dict[Tuple[str, str], int] = {}
        variant_parts: Dict[str, List[str]] = {}
        citations: List[str] = []
        path_to_cite: Dict[str, int] = {}
        used_hits: List[dict] = []

        disallow_terms: List[str] = []
        pl = canonicalize_part_key(part).lower()
        if "engine" in pl and "oil" in pl and "filter" in pl:
            disallow_terms = ["hydraulic", "fuel", "air"]
        elif "oil" in pl and "filter" in pl and "hydraulic" not in pl:
            # Treat ambiguous "oil filter" as engine oil filter by default.
            disallow_terms = ["hydraulic", "fuel", "air"]
        elif "hydraulic" in pl and "filter" in pl:
            disallow_terms = ["engine", "fuel", "air"]
        elif "fuel" in pl and "filter" in pl:
            disallow_terms = ["engine", "hydraulic", "air"]

        def _filter_pns_by_local_context(text: str, pns: List[str]) -> List[str]:
            if not disallow_terms or not text or not pns:
                return pns
            tl = text.lower()
            out: List[str] = []
            ctx_window = 220
            for pn in pns:
                pn_l = pn.lower()
                idx = tl.find(pn_l)
                if idx >= 0:
                    ctx = tl[max(0, idx - ctx_window) : min(len(tl), idx + ctx_window)]
                    if any(t in ctx for t in disallow_terms):
                        continue
                out.append(pn)
            return out

        for rec, dist in hits:
            path = rec.get("path", "") or ""
            variant = variant_from_path(path)
            pns = extract_part_numbers_near_keywords(rec.get("text", "") or "", keywords)
            pns = _filter_pns_by_local_context(rec.get("text", "") or "", pns)
            if not pns:
                continue

            if path not in path_to_cite:
                citations.append(path)
                path_to_cite[path] = len(citations)
            cite_idx = int(path_to_cite[path])

            used_hits.append({"path": path, "page_number": rec.get("page_number"), "dist": float(dist)})
            bucket = variant_parts.setdefault(variant, [])
            for pn in pns:
                if pn not in bucket:
                    bucket.append(pn)
                pn_cite.setdefault((variant, pn), cite_idx)

        if not variant_parts:
            return "", {}, [], []

        lines = []
        if model:
            lines.append(f"Model: {model}")
        lines.append(f"Part: {part}")
        lines.append("")
        lines.append("Part numbers by model variant:")
        for variant, pns in variant_parts.items():
            if not pns:
                continue
            rendered = []
            for pn in pns:
                ci = pn_cite.get((variant, pn))
                rendered.append(f"{pn} [{ci}]" if ci else pn)
            lines.append(f"- {variant}: {', '.join(rendered)}")
        lines.append("")
        lines.append("If you can provide the serial number, I can confirm the exact variant when a serial break applies.")

        # Return citations in the normal response field so the UI can show clickable paths.
        return "\n".join(lines).strip(), variant_parts, citations, used_hits

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

    # Deterministic parts extraction (prevents hallucinated part numbers).
    if parts_mode:
        model_label = structured_model or (question_models[0] if question_models else "")
        part_label = canonicalize_part_key(structured_part or infer_part_label(q), q)
        keywords = guess_part_keywords(q, part_label)

        # Some parts books label the engine oil filter as a generic "ELEMENT (... SPIN-ON)".
        # Add those cues so keyword-based extraction can still find the right row.
        pl = (part_label or "").lower()
        if "engine" in pl and "oil" in pl and "filter" in pl:
            for k in ("spin-on", "12nf"):
                if k not in keywords:
                    keywords.append(k)

        def _wants_filter_list(question: str, part_label: str) -> bool:
            ql = (question or "").lower()
            pl = (part_label or "").lower()
            if "filter" not in ql and "filter" not in pl:
                return False
            # Don't treat specific requests as a "list" query.
            specific = any(k in ql for k in ("engine oil", "hydraulic", "fuel", "air"))
            if specific:
                return False
            return ("list" in ql) or ("all" in ql and "filter" in ql) or (pl.strip() in ("filters", "filter list"))

        def _normalize_filter_label(desc: str, context_text: str) -> str:
            t = " ".join((desc or "").split())
            t = re.sub(r"\.{2,}", " ", t).strip(" -:\t")
            tl = t.lower()
            cl = (context_text or "").lower()

            # Use section context to disambiguate vague "ELEMENT (...)" rows.
            if "engine service parts" in cl and "element" in tl and (
                "spin-on" in tl or "12nf" in tl or "3.56" in tl or "3.562" in tl
            ):
                return "Engine oil filter"
            if ("filter and reservoir" in cl or "hydraulic" in cl) and "element" in tl and (
                "filter element" in tl or "micron" in tl or "cartridge" in tl or "spin-on" in tl
            ):
                return "Hydraulic oil filter"

            if "air" in tl and "filter" in tl:
                label = "Air filter"
                if "primary" in tl:
                    label += " (primary)"
                elif "secondary" in tl:
                    label += " (secondary)"
                elif "safety" in tl:
                    label += " (safety)"
                return label
            if "intake system" in cl and "element" in tl:
                if "safety" in tl:
                    return "Air filter (safety)"
                if "secondary" in tl:
                    return "Air filter (secondary)"
                if "primary" in tl:
                    return "Air filter (primary)"
                return "Air filter"

            if "fuel" in tl and "filter" in tl:
                label = "Fuel filter"
                if "pre" in tl and "separator" in tl:
                    label += " (pre-filter / water separator)"
                elif "primary" in tl:
                    label += " (primary)"
                elif "secondary" in tl:
                    label += " (secondary)"
                elif "main" in tl:
                    label += " (main)"
                return label
            if ("fuel system" in cl or "fuel" in tl or "separator" in tl) and "element" in tl:
                if "water" in tl and "separator" in tl:
                    return "Fuel filter (pre-filter / water separator)"
                if "pre-filter" in tl or ("pre" in tl and "separator" in tl):
                    return "Fuel filter (pre-filter / water separator)"
                if "main" in tl:
                    return "Fuel filter (main)"
                return "Fuel filter"

            if "hydraulic" in tl and "filter" in tl:
                return "Hydraulic oil filter"
            if "oil" in tl and "filter" in tl:
                return "Engine oil filter"
            return t[:80] if t else "Filter"

        def _extract_filter_entries(text: str) -> List[Tuple[str, str, str]]:
            if not text:
                return []
            t = " ".join(text.split())
            tl = t.lower()
            out: List[Tuple[str, str, str]] = []
            seen = set()

            def _trim_desc(desc: str) -> str:
                d = " ".join((desc or "").split()).strip()
                # Some pages append extra PN-only lines without ref/qty; avoid bleeding those into the row description.
                m = PART_NO_RE.search(d)
                if m:
                    d = d[: m.start()].strip()
                return d[:220]

            def _looks_like_filter(desc: str, ctx: str) -> bool:
                dl = (desc or "").lower()
                cl = (ctx or "").lower()
                if any(b in dl for b in (" psi", "pressure", "test port", "hydrostatic")):
                    return False
                if any(b in dl for b in ("filter head", "filter base", "filter bowl")):
                    return False
                if "filter" in dl:
                    return any(
                        k in dl
                        for k in (
                            "filter element",
                            "air filter",
                            "fuel filter",
                            "main fuel filter",
                            "pre-filter",
                            "water separator",
                            "oil filter",
                        )
                    ) or ("hydraulic" in cl)
                if "element" in dl:
                    if any(k in dl for k in ("spin-on", "micron", "cartridge", "separator", "primary", "secondary", "safety")):
                        return True
                    if "engine service parts" in cl:
                        return True
                    if "intake system" in cl:
                        return True
                    if "fuel system" in cl and ("pre" in dl or "separator" in dl or "fuel" in dl):
                        return True
                    if "hydraulic" in cl and "filter" in cl:
                        return True
                return False

            # Common structure: "REF PN QTY DESCRIPTION ..."
            ref_row_re = re.compile(
                r"\b(?P<ref>\d{1,3})\s+(?P<pn>\d{3}-\d{3,5})\s+(?P<qty>\d+)\s+(?P<desc>.+?)(?=\s+\d{1,3}\s+\d{3}-\d{3,5}\s+\d+|\Z)",
                re.IGNORECASE,
            )
            for m in ref_row_re.finditer(t):
                pn = m.group("pn")
                desc = _trim_desc(m.group("desc") or "")
                if not _looks_like_filter(desc, tl):
                    continue
                key = (pn, desc[:80])
                if key in seen:
                    continue
                seen.add(key)
                out.append((desc, pn, tl))

            # Shorthand consumables sometimes appear without REF/QTY, e.g.:
            # "194-830 ELEMENT PRIMARY 194-577 SAFETY ELEMENT"
            shorthand_re = re.compile(
                r"\b(?P<pn>\d{3}-\d{3,5})\s+(?P<desc>(?:FILTER\s+ELEMENT|(?:AIR|FUEL)\s+FILTER|MAIN\s+FUEL\s+FILTER|PRE-?FILTER\b.{0,40}?|WATER\s+SEPARATOR|ELEMENT\s+(?:PRIMARY|SECONDARY|SAFETY)|(?:PRIMARY|SECONDARY|SAFETY)\s+ELEMENT|ELEMENT\b.{0,120}?))(?=\s+\d{3}-\d{3,5}\b|\Z)",
                re.IGNORECASE,
            )
            for m in shorthand_re.finditer(t):
                pn = m.group("pn")
                desc = _trim_desc(m.group("desc") or "")
                if not _looks_like_filter(desc, tl):
                    continue
                key = (pn, desc[:80])
                if key in seen:
                    continue
                seen.add(key)
                out.append((desc, pn, tl))
            return out

        def _format_engine_oil_filter_from_model_parts(
            model: str,
            ranked_hits: List[Tuple[dict, float]],
            db_path: Path,
        ) -> tuple[str, Dict[str, List[str]], List[str], List[dict]]:
            # Pick a few likely parts PDFs first.
            candidate_paths: List[str] = []
            for rec, _dist in ranked_hits[:300]:
                p = (rec.get("path", "") or "")
                pl = p.lower()
                if any(k in pl for k in ("complete parts book", "parts manual", "parts book", "suggested parts")):
                    if p not in candidate_paths:
                        candidate_paths.append(p)
                if len(candidate_paths) >= 8:
                    break

            if not candidate_paths:
                return "", {}, [], []

            extracted: List[Tuple[dict, float]] = []
            try:
                conn = sqlite3.connect(str(db_path))
                try:
                    for p in candidate_paths:
                        rows = conn.execute(
                            """
                            SELECT page_number, text
                            FROM chunks
                            WHERE path = ?
                              AND LOWER(text) LIKE '%engine service parts%'
                              AND (LOWER(text) LIKE '%spin-on%' OR LOWER(text) LIKE '%12nf%')
                            LIMIT 120
                            """,
                            (p,),
                        ).fetchall()
                        for pn, txt in rows:
                            extracted.append(({"path": p, "page_number": int(pn or 0), "text": str(txt or "")}, 0.0))
                finally:
                    conn.close()
            except Exception:
                return "", {}, [], []

            variant_parts: Dict[str, List[str]] = {}
            citations: List[str] = []
            path_to_cite: Dict[str, int] = {}
            used_hits: List[dict] = []

            for rec, dist in extracted:
                txt = rec.get("text", "") or ""
                p = rec.get("path", "") or ""
                if not p:
                    continue
                # Prefer the PN on the actual "ELEMENT (... SPIN-ON)" row, not the next table row's PN.
                pns: List[str] = []
                for m in re.finditer(
                    r"\b\d{1,3}\s+(?P<pn>\d{3}-\d{3,5})\s+\d+\s+ELEMENT\b.{0,180}?(?:SPIN-ON|12NF)",
                    txt,
                    re.IGNORECASE,
                ):
                    pn = m.group("pn")
                    if pn and pn not in pns:
                        pns.append(pn)

                if not pns:
                    pns = extract_part_numbers_near_keywords(txt, ["oil filter"], window=180, max_pns=2)
                if not pns:
                    continue

                if p not in path_to_cite:
                    citations.append(p)
                    path_to_cite[p] = len(citations)
                used_hits.append({"path": p, "page_number": rec.get("page_number"), "dist": float(dist)})

                variant = variant_from_path(p)
                bucket = variant_parts.setdefault(variant, [])
                for pn in pns:
                    if pn not in bucket:
                        bucket.append(pn)

            if not variant_parts:
                return "", {}, [], []

            lines = []
            lines.append(f"Model: {model}")
            lines.append("Part: engine oil filter")
            lines.append("")
            lines.append("Part numbers by model variant:")
            for variant, pns in variant_parts.items():
                if not pns:
                    continue
                lines.append(f"- {variant}: {', '.join(pns)}")
            lines.append("")
            lines.append("If you can provide the serial number, I can confirm the exact variant when a serial break applies.")

            return "\n".join(lines).strip(), variant_parts, citations, used_hits

        def _format_filter_list_from_model_parts(
            model: str,
            ranked_hits: List[Tuple[dict, float]],
            db_path: Path,
        ) -> tuple[str, Dict[str, List[str]], List[str], List[dict]]:
            # Pick a few likely parts PDFs first.
            candidate_paths: List[str] = []
            for rec, _dist in ranked_hits[:300]:
                p = (rec.get("path", "") or "")
                pl = p.lower()
                if any(k in pl for k in ("complete parts book", "parts manual", "parts book", "suggested parts")):
                    if p not in candidate_paths:
                        candidate_paths.append(p)
                if len(candidate_paths) >= 8:
                    break

            if not candidate_paths:
                return "", {}, [], []

            # Pull chunks containing filter-related text from those PDFs.
            extracted: List[Tuple[dict, float]] = []
            try:
                conn = sqlite3.connect(str(db_path))
                try:
                    for p in candidate_paths:
                        rows = conn.execute(
                            """
                            SELECT page_number, text
                            FROM chunks
                            WHERE path = ? AND (
                              LOWER(text) LIKE '%filters%'
                              OR LOWER(text) LIKE '%filter element%'
                              OR LOWER(text) LIKE '%air filter%'
                              OR LOWER(text) LIKE '%fuel filter%'
                              OR LOWER(text) LIKE '%oil filter%'
                              OR LOWER(text) LIKE '%water separator%'
                              OR LOWER(text) LIKE '%engine service parts%'
                              OR LOWER(text) LIKE '%intake system%'
                              OR LOWER(text) LIKE '%fuel system%'
                              OR LOWER(text) LIKE '%filter and reservoir%'
                            )
                            LIMIT 160
                            """,
                            (p,),
                        ).fetchall()
                        for pn, txt in rows:
                            extracted.append(({"path": p, "page_number": int(pn or 0), "text": str(txt or "")}, 0.0))
                finally:
                    conn.close()
            except Exception:
                return "", {}, [], []

            citations: List[str] = []
            path_to_cite: Dict[str, int] = {}
            used_hits: List[dict] = []
            label_map: Dict[str, List[str]] = {}
            allowed_labels = {
                "Air filter",
                "Air filter (primary)",
                "Air filter (secondary)",
                "Air filter (safety)",
                "Fuel filter",
                "Fuel filter (primary)",
                "Fuel filter (secondary)",
                "Fuel filter (main)",
                "Fuel filter (pre-filter / water separator)",
                "Hydraulic oil filter",
                "Engine oil filter",
            }
            for rec, dist in extracted:
                p = rec.get("path", "") or ""
                if p not in path_to_cite:
                    citations.append(p)
                    path_to_cite[p] = len(citations)
                used_hits.append({"path": p, "page_number": rec.get("page_number"), "dist": float(dist)})
                for desc, pn, ctx in _extract_filter_entries(rec.get("text", "") or ""):
                    label = _normalize_filter_label(desc, ctx)
                    if label not in allowed_labels:
                        continue
                    bucket = label_map.setdefault(label, [])
                    if pn not in bucket:
                        bucket.append(pn)

            if not label_map:
                return "", {}, [], []

            lines = []
            lines.append(f"Model: {model}")
            lines.append("Part: filters")
            lines.append("")
            lines.append("Filters found:")
            for label in sorted(label_map.keys()):
                pns = label_map[label]
                if not pns:
                    continue
                lines.append(f"- {label}: {', '.join(pns)}")
            lines.append("")
            lines.append("If you can provide the serial number, I can confirm the exact variant when a serial break applies.")

            # Expose a simple variant_parts mapping for the UI/feedback (single variant).
            variant_parts = {variant_from_path(candidate_paths[0]): sorted({pn for ps in label_map.values() for pn in ps})}
            return "\n".join(lines).strip(), variant_parts, citations, used_hits

        if model_label and _wants_filter_list(q, part_label):
            fl_answer, fl_variant_parts, fl_sources, fl_hits = _format_filter_list_from_model_parts(
                model_label, ranked_hits, db_path
            )
            if fl_answer and fl_sources:
                return {
                    "answer": fl_answer,
                    "sources": fl_sources,
                    "used_vision": False,
                    "hits": fl_hits,
                    "variant_parts": fl_variant_parts,
                    "feedback_used": False,
                }

        if model_label and ("engine" in pl and "oil" in pl and "filter" in pl):
            eo_answer, eo_vp, eo_sources, eo_hits = _format_engine_oil_filter_from_model_parts(
                model_label, ranked_hits, db_path
            )
            if eo_answer and eo_sources:
                return {
                    "answer": eo_answer,
                    "sources": eo_sources,
                    "used_vision": False,
                    "hits": eo_hits,
                    "variant_parts": eo_vp,
                    "feedback_used": False,
                }
        # Search more than just the final top_k to ensure we find the relevant "oil filter element" row.
        direct_answer, direct_variant_parts, direct_citations, direct_hits = format_parts_from_hits(
            model_label, part_label, ranked_hits[:2000], keywords
        )
        if direct_answer and direct_variant_parts and direct_citations:
            return {
                "answer": direct_answer,
                "sources": direct_citations,
                "used_vision": False,
                "hits": direct_hits,
                "variant_parts": direct_variant_parts,
                "feedback_used": False,
            }

        # If the user is asking for a part (not diagnosis), avoid slow/hallucinated LLM fallback.
        wants_part_number = bool(structured_part) or ("part" in q.lower())
        if wants_part_number and model_label:
            # Fallback: keyword search within likely parts PDFs to find the relevant table row.
            candidate_paths: List[str] = []
            for rec, _dist in ranked_hits[:200]:
                p = (rec.get("path", "") or "")
                pl = p.lower()
                if any(k in pl for k in ("complete parts book", "parts manual", "parts book", "suggested parts")):
                    if p not in candidate_paths:
                        candidate_paths.append(p)
                if len(candidate_paths) >= 6:
                    break

            extra_hits: List[Tuple[dict, float]] = []
            if candidate_paths:
                try:
                    conn = sqlite3.connect(str(db_path))
                    try:
                        like_terms = [k for k in keywords if k and k not in ("filter", "filter element")]
                        # Add a minimal fallback term for common oil-filter wording.
                        if not like_terms and "engine oil" in q.lower():
                            like_terms = ["oil filter"]
                        patterns = [f"%{t.lower()}%" for t in like_terms[:6]]

                        for p in candidate_paths:
                            if not patterns:
                                break
                            ors = " OR ".join("LOWER(text) LIKE ?" for _ in patterns)
                            sql = f"SELECT page_number, text FROM chunks WHERE path = ? AND ({ors}) LIMIT 60"
                            for pn, txt in conn.execute(sql, tuple([p] + patterns)):
                                extra_hits.append(
                                    (
                                        {"path": p, "page_number": int(pn or 0), "text": str(txt or "")},
                                        0.0,
                                    )
                                )
                    finally:
                        conn.close()
                except Exception:
                    extra_hits = []

            if extra_hits:
                extra_answer, extra_vp, extra_sources, extra_used_hits = format_parts_from_hits(
                    model_label, part_label, extra_hits, keywords
                )
                if extra_answer and extra_vp and extra_sources:
                    return {
                        "answer": extra_answer,
                        "sources": extra_sources,
                        "used_vision": False,
                        "hits": extra_used_hits,
                        "variant_parts": extra_vp,
                        "feedback_used": False,
                    }

            # Show the top likely manuals so the user can refine the query.
            seen = set()
            fallback_sources = []
            fallback_hits = []
            for rec, dist in ranked_hits[:20]:
                path = rec.get("path", "") or ""
                if not path or path in seen:
                    continue
                seen.add(path)
                fallback_sources.append(path)
                fallback_hits.append({"path": path, "page_number": rec.get("page_number"), "dist": float(dist)})
                if len(fallback_sources) >= 6:
                    break
            return {
                "answer": (
                    f"Model: {model_label}\n"
                    f"Part: {part_label}\n\n"
                    "I couldn't find a part number for this request in the retrieved manual text.\n"
                    "Try `Request type = Parts lookup` and use a short description like `engine oil filter` / `oil filter element`,\n"
                    "or tell me the engine make/model (Kohler/Kubota/etc.) so I can match the right filter.\n"
                ).strip(),
                "sources": fallback_sources,
                "used_vision": False,
                "hits": fallback_hits,
                "variant_parts": {},
                "feedback_used": False,
            }

    answer = generate(prompt, base_url, chat_model, session, timeout_s=generate_timeout_s)

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
                        vision_answer = chat_vision(
                            vision_prompt, images_b64, base_url, vision_model, session, timeout_s=vision_timeout_s
                        )
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
