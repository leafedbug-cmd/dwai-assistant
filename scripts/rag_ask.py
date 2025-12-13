import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
import hnswlib
import fitz  # PyMuPDF
import base64


def load_index(index_dir: Path):
    index_path = index_dir / "index.bin"
    meta_path = index_dir / "meta.jsonl"
    cfg_path = index_dir / "config.json"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found. Run rag_reindex.py first.")
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Read meta into memory (id order matches index ids)
    meta: List[dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    if not meta:
        raise ValueError("Metadata is empty. Rebuild the index.")
    return index_path, meta, cfg


def embed(text: str, base_url: str, model: str, session: requests.Session) -> np.ndarray:
    url = base_url.rstrip("/") + "/api/embeddings"
    resp = session.post(url, json={"model": model, "prompt": text}, timeout=120)
    resp.raise_for_status()
    vec = np.array(resp.json()["embedding"], dtype=np.float32)
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec


def generate(prompt: str, base_url: str, model: str, session: requests.Session) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    resp = session.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def generate_vision(
    prompt: str, images_b64: List[str], base_url: str, model: str, session: requests.Session
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    resp = session.post(
        url,
        json={"model": model, "prompt": prompt, "images": images_b64, "stream": False},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


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


def extract_model_tokens(question: str) -> Tuple[List[str], List[str]]:
    """
    Returns (model_variants, qualifiers) where:
      - model_variants are normalized strings likely to appear in paths (e.g., "rt45", "rt-45", "rt 45")
      - qualifiers include things like "tier 4", "stage v" if present in the question
    """
    q = question.lower()
    qualifiers: List[str] = []
    for qual in ("tier 4", "tier 4f", "tier 4i", "tier 3", "stage v", "stage vt", "stage v tier"):
        if qual in q:
            qualifiers.append(qual)

    # Capture tokens like RT45, JT2020, SK1550, etc (allow optional hyphen/space).
    raw = set()
    for m in re.finditer(r"\b([a-z]{1,4})\s*[-]?\s*(\d{1,4}[a-z]?)\b", q, flags=re.IGNORECASE):
        raw.add(f"{m.group(1)}{m.group(2)}".lower())

    variants: List[str] = []
    for token in sorted(raw):
        # Basic variants that appear in folder names.
        letters = re.match(r"^[a-z]+", token).group(0) if re.match(r"^[a-z]+", token) else token
        digits = token[len(letters) :]
        variants.extend([token, f"{letters}-{digits}", f"{letters} {digits}"])
    # de-dupe preserving order
    out: List[str] = []
    seen = set()
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out, qualifiers


def classify_intent(question: str) -> str:
    q = question.lower()
    parts_words = ("part", "parts", "assembly", "assy", "pn", "part number", "item", "callout", "exploded", "diagram")
    specs_words = ("quart", "quarts", "oil", "capacity", "spec", "specs", "torque", "interval", "maintenance", "fluid")
    if any(w in q for w in parts_words) and not any(w in q for w in specs_words):
        return "parts"
    if any(w in q for w in specs_words) and not any(w in q for w in parts_words):
        return "specs"
    # ambiguous
    return "general"


def expand_question(question: str) -> str:
    models, qualifiers = extract_model_tokens(question)
    intent = classify_intent(question)
    additions: List[str] = []
    if models:
        additions.append("model: " + " ".join(models[:3]))
    if qualifiers:
        additions.append(" ".join(qualifiers))
    if intent == "parts":
        additions.append("parts manual complete parts book parts list item number callout assembly part number")
    elif intent == "specs":
        additions.append("operator manual specifications capacity maintenance interval")
    else:
        additions.append("service manual operator manual parts manual")
    return question.strip() + "\n\nQuery hints: " + " | ".join(additions)


def path_boost(path: str, model_variants: List[str], qualifiers: List[str], intent: str) -> float:
    p = path.lower()
    boost = 0.0

    if model_variants:
        if any(v in p for v in model_variants):
            boost += 3.0
        elif any(v.replace("-", "").replace(" ", "") in p.replace("-", "").replace(" ", "") for v in model_variants):
            boost += 2.0

    if qualifiers and any(q in p for q in qualifiers):
        boost += 1.5

    if intent == "parts":
        if "parts" in p or "complete parts book" in p:
            boost += 1.5
        if "suggested parts stocking" in p:
            boost += 0.5
        if "operator" in p:
            boost -= 0.5
    elif intent == "specs":
        if "operator" in p or "maintenance" in p:
            boost += 1.5
        if "parts" in p:
            boost -= 0.5

    # Penalize commonly irrelevant telemetry docs for parts/spec questions.
    if any(bad in p for bad in ("telematics", "clm", "gps")):
        boost -= 1.5
    return boost


def rerank_hits(question: str, hits: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
    model_variants, qualifiers = extract_model_tokens(question)
    intent = classify_intent(question)
    q = question.lower()
    scored = []
    for rec, dist in hits:
        p = (rec.get("path", "") or "").lower()
        # Hard filter noisy telemetry docs unless the user is explicitly asking about them.
        if ("telematics" in p or "clm" in p or "gps" in p) and not any(w in q for w in ("telematics", "gps", "clm")):
            continue
        b = path_boost(rec.get("path", ""), model_variants, qualifiers, intent)
        scored.append((rec, dist, b))
    if model_variants:
        # If we found any model-specific candidates, drop non-matching paths to keep results on-topic.
        matching = [x for x in scored if any(v in (x[0].get("path", "") or "").lower() for v in model_variants)]
        if matching:
            scored = matching
    # Lower distance is better; higher boost is better.
    scored.sort(key=lambda x: (x[1] - 0.08 * x[2], x[1]))
    return [(r, d) for r, d, _b in scored]


def should_use_vision(question: str, hits: List[tuple]) -> bool:
    q = question.lower()
    keyword_triggers = [
        "assembly",
        "part number",
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

    cfg = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(
                f"Config file not found: {config_path}\n"
                "Create it with:\n"
                "  Copy-Item .\\scripts\\rag_config.example.json .\\scripts\\rag_config.json",
                file=sys.stderr,
            )
        else:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))

    index_dir = Path(args.index_dir or cfg.get("index_dir", Path("data/rag").resolve()))
    answer, citations = answer_question(
        question,
        config=cfg,
        index_dir=index_dir,
        force_vision=args.vision,
        disable_vision=args.no_vision,
    )

    print(answer)
    if citations:
        print("\nSources:")
        for i, p in enumerate(citations, start=1):
            print(f"[{i}] {p}")
    return 0


def answer_question(
    question: str,
    *,
    config: dict,
    index_dir: Path,
    force_vision: bool = False,
    disable_vision: bool = False,
) -> Tuple[str, List[str]]:
    docs_root = Path(config.get("docs_root", Path("docs/service-documents").resolve()))
    base_url = config.get("ollama_base_url", "http://localhost:11434")
    embed_model = config.get("embedding_model", "nomic-embed-text")
    chat_model = config.get("chat_model", "qwen3:8b")
    vision_model = config.get("vision_model", "llava:7b")
    vision_max_pages = int(config.get("vision_max_pages", 3))
    top_k = int(config.get("top_k", 4))
    max_context_chars = int(config.get("max_context_chars", 12000))

    index_path, meta, stored_cfg = load_index(index_dir)
    if stored_cfg:
        base_url = stored_cfg.get("ollama_base_url", base_url)
        embed_model = stored_cfg.get("embedding_model", embed_model)

    session = requests.Session()
    expanded = expand_question(question)
    v1 = embed(question, base_url, embed_model, session)
    v2 = embed(expanded, base_url, embed_model, session)
    q_vec = v1 + v2
    q_vec /= (np.linalg.norm(q_vec) + 1e-8)

    dim = q_vec.shape[0]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(str(index_path))
    idx.set_ef(50)

    search_k = max(top_k * 12, 48)
    labels, distances = idx.knn_query(q_vec, k=search_k)
    hits: List[Tuple[dict, float]] = []
    for lab, dist in zip(labels[0].tolist(), distances[0].tolist()):
        if 0 <= lab < len(meta):
            hits.append((meta[lab], float(dist)))
    hits = rerank_hits(question, hits)
    hits = hits[: max(top_k * 4, top_k)]

    context_blocks: List[str] = []
    used = 0
    citations: List[str] = []
    for rec, _dist in hits[:top_k]:
        pn = rec.get("page_number")
        loc = f"{rec['path']}" + (f":p{pn}" if pn else "")
        block = f"[{len(citations)+1}] {rec['path']}\n{rec['text']}\n"
        if used + len(block) > max_context_chars:
            break
        context_blocks.append(block)
        used += len(block)
        citations.append(loc)

    context = "\n".join(context_blocks)
    prompt = (
        "You are DWAI Assistant.\n"
        "Write like a helpful text message: concise, direct, and practical.\n"
        "Use the CONTEXT from service documents. If the answer isn't in context, say what to check next.\n"
        "Ask at most ONE clarifying question if needed.\n"
        "Cite sources like [1], [2] matching the context blocks.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:"
    )

    answer = generate(prompt, base_url, chat_model, session)

    use_vision = force_vision or (not disable_vision and should_use_vision(question, hits[:top_k]))
    if use_vision and hits:
        primary_path = hits[0][0].get("path")
        pages: List[int] = []
        seen = set()
        for rec, _dist in hits:
            if primary_path and rec.get("path") != primary_path:
                continue
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
                    vision_prompt = (
                        "You are DWAI Assistant. The images are pages from equipment manuals.\n"
                        "If this is a parts diagram/exploded view, map callout numbers to part numbers/names.\n"
                        "If this is a spec/maintenance page, extract the requested value precisely.\n"
                        "If unsure, say what page/section is needed.\n\n"
                        f"QUESTION:\n{question}\n"
                    )
                    try:
                        vision_answer = generate_vision(vision_prompt, images_b64, base_url, vision_model, session)
                        if vision_answer:
                            answer = vision_answer
                    except Exception:
                        pass

    return answer, citations


if __name__ == "__main__":
    raise SystemExit(main())
