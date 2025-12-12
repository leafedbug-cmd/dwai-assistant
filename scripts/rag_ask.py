import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

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

    # Load config defaults
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
    docs_root = Path(cfg.get("docs_root", Path("docs/service-documents").resolve()))
    index_dir = Path(
        args.index_dir
        or cfg.get("index_dir", Path("data/rag").resolve())
    )
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    embed_model = cfg.get("embedding_model", "nomic-embed-text")
    chat_model = cfg.get("chat_model", "qwen3:8b")
    vision_model = cfg.get("vision_model", "qwen2.5-vl:7b")
    vision_max_pages = int(cfg.get("vision_max_pages", 3))
    top_k = int(cfg.get("top_k", 4))
    max_context_chars = int(cfg.get("max_context_chars", 12000))

    index_path, meta, stored_cfg = load_index(index_dir)
    if stored_cfg:
        base_url = stored_cfg.get("ollama_base_url", base_url)
        embed_model = stored_cfg.get("embedding_model", embed_model)

    session = requests.Session()
    q_vec = embed(question, base_url, embed_model, session)

    # Determine dim from embedding
    dim = q_vec.shape[0]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(str(index_path))
    idx.set_ef(50)

    labels, distances = idx.knn_query(q_vec, k=top_k)
    hits = []
    for lab, dist in zip(labels[0].tolist(), distances[0].tolist()):
        if 0 <= lab < len(meta):
            hits.append((meta[lab], float(dist)))

    context_blocks = []
    used = 0
    citations = []
    for rec, dist in hits:
        block = f"[{len(citations)+1}] {rec['path']}\n{rec['text']}\n"
        if used + len(block) > max_context_chars:
            break
        context_blocks.append(block)
        used += len(block)
        citations.append(rec["path"])

    context = "\n".join(context_blocks)
    prompt = (
        "You are DWAI Assistant. Answer using the CONTEXT from service documents.\n"
        "If the answer isn't in context, say so.\n"
        "Cite sources like [1], [2] matching the context blocks.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:"
    )

    answer = generate(prompt, base_url, chat_model, session)

    use_vision = args.vision or (not args.no_vision and should_use_vision(question, hits))
    if use_vision and hits:
        # Collect top unique pages from hits, if available.
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
            pdf_paths = []
            for rec, _dist in hits:
                pdf_paths.append(docs_root / rec["path"])
            # Use the first hit's PDF for page rendering.
            pdf_path = pdf_paths[0]
            if pdf_path.exists():
                images_b64 = render_pdf_pages(pdf_path, pages)
                if images_b64:
                    vision_prompt = (
                        "You are DWAI Assistant. The images are pages from a parts manual.\n"
                        "Use the diagrams, callout numbers, and parts lists to answer.\n"
                        "If you can, map callout numbers to part numbers and names.\n\n"
                        f"QUESTION:\n{question}\n"
                    )
                    try:
                        vision_answer = chat_vision(vision_prompt, images_b64, base_url, vision_model, session)
                        if vision_answer:
                            answer = vision_answer
                    except Exception:
                        pass

    print(answer)
    if citations:
        print("\nSources:")
        for i, p in enumerate(citations, start=1):
            print(f"[{i}] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
