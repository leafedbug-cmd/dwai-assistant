import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import requests
from pypdf import PdfReader
from tqdm import tqdm
import hnswlib


def load_config(path: str | None) -> dict:
    default = {
        "docs_root": str(Path("docs/service-documents").resolve()),
        "index_dir": str(Path("data/rag").resolve()),
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "embedding_model": "nomic-embed-text",
        "ollama_base_url": "http://localhost:11434",
    }
    if not path:
        return default
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    return {**default, **user}


def iter_pdfs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.pdf"):
        if p.is_file():
            yield p


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number_1_based, text).
    If extraction fails, returns empty list.
    """
    try:
        reader = PdfReader(str(pdf_path))
        out: List[Tuple[int, str]] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                out.append((i + 1, text))
        return out
    except Exception:
        return []


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


def embed_texts(
    texts: List[str], base_url: str, model: str, session: requests.Session
) -> np.ndarray:
    url = base_url.rstrip("/") + "/api/embeddings"
    vectors: List[List[float]] = []
    for t in texts:
        resp = session.post(url, json={"model": model, "prompt": t}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        vectors.append(data["embedding"])
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a local RAG index over service PDFs.")
    ap.add_argument("--config", help="Path to rag_config.json")
    ap.add_argument("--dry-run", action="store_true", help="Scan and count chunks only.")
    ap.add_argument(
        "--batch-size", type=int, default=16, help="Embedding batch size (sequential calls)."
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    docs_root = Path(cfg["docs_root"])
    index_dir = Path(cfg["index_dir"])
    chunk_size = int(cfg["chunk_size"])
    overlap = int(cfg["chunk_overlap"])
    embed_model = cfg["embedding_model"]
    base_url = cfg["ollama_base_url"]

    if not docs_root.exists():
        print(f"Docs root not found: {docs_root}", file=sys.stderr)
        return 2

    pdfs = list(iter_pdfs(docs_root))
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 1

    print(f"Found {len(pdfs)} PDFs. Counting chunks...")
    total_chunks = 0
    per_pdf_chunks: List[Tuple[Path, List[Tuple[int, str]]]] = []
    for pdf in tqdm(pdfs, desc="Counting"):
        pages = extract_pdf_pages(pdf)
        page_chunks: List[Tuple[int, str]] = []
        for page_num, page_text in pages:
            chunks = chunk_text(page_text, chunk_size, overlap)
            for c in chunks:
                page_chunks.append((page_num, c))
        if page_chunks:
            per_pdf_chunks.append((pdf, page_chunks))
            total_chunks += len(page_chunks)

    print(f"Total chunks to index: {total_chunks}")
    if args.dry_run:
        return 0

    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "meta.jsonl"
    index_path = index_dir / "index.bin"
    cfg_path = index_dir / "config.json"

    if total_chunks == 0:
        print("No extractable text found in PDFs.", file=sys.stderr)
        return 1

    session = requests.Session()
    dim_probe = embed_texts(["probe"], base_url, embed_model, session).shape[1]

    idx = hnswlib.Index(space="cosine", dim=dim_probe)
    idx.init_index(max_elements=total_chunks, ef_construction=200, M=64)
    idx.set_ef(50)

    current_id = 0
    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for pdf, chunks in tqdm(per_pdf_chunks, desc="Embedding"):
            rel = str(pdf.relative_to(docs_root)).replace("\\", "/")
            for i in range(0, len(chunks), args.batch_size):
                batch = chunks[i : i + args.batch_size]
                batch_texts = [t for _, t in batch]
                vecs = embed_texts(batch_texts, base_url, embed_model, session)
                ids = np.arange(current_id, current_id + len(batch))
                idx.add_items(vecs, ids)
                for j, (page_num, chunk) in enumerate(batch):
                    rec = {
                        "id": int(ids[j]),
                        "path": rel,
                        "chunk_index": int(i + j),
                        "page_number": int(page_num),
                        "text": chunk,
                    }
                    meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                current_id += len(batch)

    idx.save_index(str(index_path))
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Index written to: {index_path}")
    print(f"Metadata written to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
