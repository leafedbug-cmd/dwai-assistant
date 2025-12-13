import argparse
import json
import os
import sys
import sqlite3
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import requests
from tqdm import tqdm
import hnswlib
import fitz  # PyMuPDF
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress noisy MuPDF parse warnings from some PDFs.
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass


RE_LONG_RUN = re.compile(r"\S{120,}")


def normalize_for_embedding(text: str, max_chars: int) -> str:
    """
    Ollama embedding models have a context limit. Some PDFs contain long,
    unbroken strings (tables/IDs) that can blow up tokenization. This:
    - normalizes whitespace,
    - inserts spaces into very long non-space runs,
    - truncates to max_chars as a last resort.
    """
    t = " ".join((text or "").split())
    if not t:
        return ""

    def _break(m: re.Match) -> str:
        s = m.group(0)
        # Insert spaces every 40 chars to avoid huge single tokens.
        return " ".join(s[i : i + 40] for i in range(0, len(s), 40))

    t = RE_LONG_RUN.sub(_break, t)
    if len(t) > max_chars:
        t = t[:max_chars]
    return t


def load_config(path: str | None) -> dict:
    default = {
        "docs_root": str(Path("docs/service-documents").resolve()),
        "index_dir": str(Path("data/rag").resolve()),
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "embedding_model": "nomic-embed-text",
        "ollama_base_url": "http://localhost:11434",
        "embedding_max_chars": 6000,
    }
    if not path:
        return default
    if not os.path.exists(path):
        print(
            f"Config file not found: {path}\n"
            "Create it with:\n"
            "  Copy-Item .\\scripts\\rag_config.example.json .\\scripts\\rag_config.json",
            file=sys.stderr,
        )
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
    Uses PyMuPDF for speed; returns empty list on failure.
    """
    out: List[Tuple[int, str]] = []
    try:
        doc = fitz.open(pdf_path)
        try:
            for i in range(doc.page_count):
                page = doc.load_page(i)
                text = page.get_text("text") or ""
                if text.strip():
                    out.append((i + 1, text))
        finally:
            doc.close()
    except Exception:
        return []
    return out


def process_pdf_for_chunks(args: Tuple[str, str, int, int]) -> Tuple[str, List[Tuple[int, str]]]:
    pdf_str, docs_root_str, chunk_size, overlap = args
    pdf = Path(pdf_str)
    docs_root = Path(docs_root_str)
    pages = extract_pdf_pages(pdf)
    page_chunks: List[Tuple[int, str]] = []
    for page_num, page_text in pages:
        chunks = chunk_text(page_text, chunk_size, overlap)
        for c in chunks:
            page_chunks.append((page_num, c))
    rel = str(pdf.relative_to(docs_root)).replace("\\", "/")
    return rel, page_chunks


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
    texts: List[str],
    base_url: str,
    model: str,
    session: requests.Session,
    *,
    max_chars: int,
) -> np.ndarray:
    """
    Returns L2-normalized embeddings (float32).

    Prefer the batch endpoint (/api/embed) to reduce HTTP overhead.
    Fall back to the legacy single-prompt endpoint (/api/embeddings).
    """
    url_embed = base_url.rstrip("/") + "/api/embed"
    url_legacy = base_url.rstrip("/") + "/api/embeddings"

    prepared = [normalize_for_embedding(t, max_chars=max_chars) for t in texts]
    vectors: List[List[float]] = []

    def _post_embed(batch: List[str]) -> List[List[float]]:
        resp = session.post(url_embed, json={"model": model, "input": batch}, timeout=600)
        if resp.status_code == 404:
            out: List[List[float]] = []
            for t in batch:
                r = session.post(url_legacy, json={"model": model, "prompt": t}, timeout=600)
                r.raise_for_status()
                out.append(r.json()["embedding"])
            return out
        if resp.ok:
            data = resp.json()
            if "embeddings" in data and isinstance(data["embeddings"], list):
                return data["embeddings"]
            if "data" in data and isinstance(data["data"], list) and data["data"]:
                return [row["embedding"] for row in data["data"] if "embedding" in row]
            raise ValueError(f"Unexpected /api/embed response shape: {list(data.keys())}")

        # Handle context-length errors by splitting batch, then truncating single items.
        try:
            err = resp.json().get("error", "") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        except Exception:
            err = resp.text
        if "exceeds the context length" in (err or "").lower():
            if len(batch) > 1:
                mid = len(batch) // 2
                return _post_embed(batch[:mid]) + _post_embed(batch[mid:])
            # single item: be more aggressive
            t = batch[0]
            shorter = normalize_for_embedding(t, max_chars=max(500, int(max_chars * 0.6)))
            if shorter == t:
                shorter = t[: max(500, int(len(t) * 0.6))]
            resp2 = session.post(url_embed, json={"model": model, "input": [shorter]}, timeout=600)
            if resp2.ok:
                data = resp2.json()
                if "embeddings" in data and data["embeddings"]:
                    return [data["embeddings"][0]]
                if "data" in data and data["data"]:
                    return [data["data"][0]["embedding"]]
            resp2.raise_for_status()
        resp.raise_for_status()
        raise RuntimeError("Unreachable")

    vectors = _post_embed(prepared)
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms


def open_meta_db(index_dir: Path) -> sqlite3.Connection:
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "meta.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
          id INTEGER PRIMARY KEY,
          path TEXT NOT NULL,
          chunk_index INTEGER NOT NULL,
          page_number INTEGER NOT NULL,
          text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          size INTEGER NOT NULL,
          mtime_ns INTEGER NOT NULL,
          max_id INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS state (
          k TEXT PRIMARY KEY,
          v TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_files_max_id ON files(max_id);")
    return conn


def migrate_jsonl_to_sqlite(index_dir: Path, conn: sqlite3.Connection) -> None:
    """
    One-time migration for older indexes that only have meta.jsonl.
    This enables incremental updates without re-embedding everything.
    """
    jsonl_path = index_dir / "meta.jsonl"
    if not jsonl_path.exists():
        return

    existing = conn.execute("SELECT COUNT(1) FROM chunks").fetchone()[0]
    if existing:
        return

    print("Migrating meta.jsonl -> meta.sqlite (one-time)...")
    with open(jsonl_path, "r", encoding="utf-8") as f, conn:
        batch = []
        max_id = -1
        per_path: dict[str, int] = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = int(rec["id"])
            max_id = max(max_id, rid)
            rpath = str(rec["path"])
            per_path[rpath] = max(per_path.get(rpath, -1), rid + 1)
            batch.append(
                (
                    rid,
                    rpath,
                    int(rec.get("chunk_index", 0)),
                    int(rec.get("page_number", 0)),
                    str(rec.get("text", "") or ""),
                )
            )
            if len(batch) >= 1000:
                conn.executemany(
                    "INSERT OR REPLACE INTO chunks(id,path,chunk_index,page_number,text) VALUES (?,?,?,?,?)",
                    batch,
                )
                batch.clear()
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks(id,path,chunk_index,page_number,text) VALUES (?,?,?,?,?)",
                batch,
            )

        # Seed state so incremental runs can skip already-indexed files.
        if max_id >= 0:
            conn.execute(
                "INSERT OR REPLACE INTO state(k,v) VALUES('last_saved_id', ?)",
                (str(max_id + 1),),
            )

        # Best-effort: mark files as indexed (unknown size/mtime at migration time).
        # These rows will be overwritten with real signatures as files are reprocessed.
        for p, mid in per_path.items():
            conn.execute(
                "INSERT OR IGNORE INTO files(path,size,mtime_ns,max_id) VALUES (?,?,?,?)",
                (p, 0, 0, int(mid)),
            )


def get_indexed_files(conn: sqlite3.Connection) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    for path, size, mtime_ns, max_id in conn.execute("SELECT path,size,mtime_ns,max_id FROM files"):
        out[str(path)] = (int(size), int(mtime_ns), int(max_id))
    return out


def get_current_chunk_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(id) FROM chunks").fetchone()
    if not row or row[0] is None:
        return 0
    return int(row[0]) + 1


def get_state_int(conn: sqlite3.Connection, key: str, default: int) -> int:
    row = conn.execute("SELECT v FROM state WHERE k = ?", (key,)).fetchone()
    if not row or row[0] is None:
        return default
    try:
        return int(row[0])
    except Exception:
        return default


def set_state_int(conn: sqlite3.Connection, key: str, value: int) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO state(k,v) VALUES(?,?)",
            (key, str(int(value))),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a local RAG index over service PDFs.")
    ap.add_argument("--config", help="Path to rag_config.json")
    ap.add_argument("--dry-run", action="store_true", help="Scan and count chunks only.")
    ap.add_argument(
        "--batch-size", type=int, default=16, help="Embedding batch size (sequential calls)."
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel workers for PDF extraction/counting.",
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild from scratch (ignores any existing index/metadata).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    docs_root = Path(cfg["docs_root"])
    index_dir = Path(cfg["index_dir"])
    chunk_size = int(cfg["chunk_size"])
    overlap = int(cfg["chunk_overlap"])
    embed_model = cfg["embedding_model"]
    base_url = cfg["ollama_base_url"]
    embed_max_chars = int(cfg.get("embedding_max_chars", 6000))
    embed_max_chars = max(embed_max_chars, chunk_size)

    if not docs_root.exists():
        print(f"Docs root not found: {docs_root}", file=sys.stderr)
        return 2

    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "meta.jsonl"
    index_path = index_dir / "index.bin"
    cfg_path = index_dir / "config.json"
    db_conn = open_meta_db(index_dir)
    migrate_jsonl_to_sqlite(index_dir, db_conn)

    session = requests.Session()
    dim_probe = embed_texts(["probe"], base_url, embed_model, session, max_chars=embed_max_chars).shape[1]

    idx = hnswlib.Index(space="cosine", dim=dim_probe)
    existing_count = 0
    indexed_files: dict[str, tuple[int, int, int]] = {}

    if index_path.exists() and cfg_path.exists() and not args.rebuild:
        try:
            stored_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            mismatches = []
            for k in ("chunk_size", "chunk_overlap", "embedding_model"):
                if str(stored_cfg.get(k)) != str(cfg.get(k)):
                    mismatches.append(f"{k}: stored={stored_cfg.get(k)} current={cfg.get(k)}")
            if mismatches:
                print(
                    "Config changed since last build; incremental indexing would mix incompatible chunking/embeddings.\n"
                    + "\n".join(f"- {m}" for m in mismatches)
                    + "\n\nRe-run with --rebuild.",
                    file=sys.stderr,
                )
                return 3
        except Exception:
            # If we can't read stored config, proceed (user can force --rebuild if needed).
            pass
    if index_path.exists() and not args.rebuild:
        try:
            idx.load_index(str(index_path))
            existing_count = idx.get_current_count()
            indexed_files = get_indexed_files(db_conn)
            db_count = get_current_chunk_count(db_conn)
            if db_count > existing_count:
                with db_conn:
                    db_conn.execute("DELETE FROM chunks WHERE id >= ?", (existing_count,))
                print(f"Trimmed metadata to match index count ({existing_count}).")
            set_state_int(db_conn, "last_saved_id", existing_count)
        except Exception as e:
            print(f"Failed to load existing index, rebuilding: {e}", file=sys.stderr)
            existing_count = 0
            indexed_files = {}

    pdf_paths = list(iter_pdfs(docs_root))
    if not pdf_paths:
        print("No PDFs found.", file=sys.stderr)
        return 1

    last_saved_id = get_state_int(db_conn, "last_saved_id", existing_count)
    if last_saved_id != existing_count:
        # Keep state aligned with what we actually loaded from disk.
        last_saved_id = existing_count
        set_state_int(db_conn, "last_saved_id", existing_count)

    def rel_and_sig(pdf_path: Path) -> tuple[str, tuple[int, int]]:
        rel = str(pdf_path.relative_to(docs_root)).replace("\\", "/")
        st = pdf_path.stat()
        return rel, (int(st.st_size), int(st.st_mtime_ns))

    # If we migrated from meta.jsonl, files table may have placeholder signatures (0/0).
    # Fill them in without re-embedding, so incremental runs can correctly skip unchanged PDFs.
    if existing_count and any(v[0] == 0 and v[1] == 0 for v in indexed_files.values()):
        rel_to_sig: dict[str, tuple[int, int]] = {}
        for p in pdf_paths:
            try:
                rel, sig = rel_and_sig(p)
                rel_to_sig[rel] = sig
            except Exception:
                continue
        with db_conn:
            for rel, (size, mtime_ns, max_id) in list(indexed_files.items()):
                if size == 0 and mtime_ns == 0 and rel in rel_to_sig:
                    ns = rel_to_sig[rel]
                    db_conn.execute(
                        "UPDATE files SET size = ?, mtime_ns = ? WHERE path = ?",
                        (int(ns[0]), int(ns[1]), rel),
                    )
        indexed_files = get_indexed_files(db_conn)

    to_process: List[Path] = []
    skipped_same = 0
    skipped_changed = 0
    for p in pdf_paths:
        rel, sig = rel_and_sig(p)
        if existing_count == 0:
            to_process.append(p)
            continue
        old = indexed_files.get(rel)
        if old is None:
            to_process.append(p)
            continue
        old_size, old_mtime_ns, old_max_id = old
        if old_size == 0 and old_mtime_ns == 0:
            # Migrated entry without a real signature; reprocess once to stamp it.
            to_process.append(p)
            continue
        if old_max_id <= last_saved_id and (old_size, old_mtime_ns) == sig:
            skipped_same += 1
            continue
        if (old_size, old_mtime_ns) != sig:
            skipped_changed += 1
            continue
        # Not fully saved previously; reprocess.
        to_process.append(p)

    if existing_count:
        print(f"Found {len(pdf_paths)} PDFs. Incremental mode: will (re)process {len(to_process)} PDFs.")
        if skipped_same:
            print(f"Incremental mode: skipping unchanged PDFs: {skipped_same}")
        if skipped_changed:
            print(
                f"Incremental mode: detected changed PDFs: {skipped_changed}. "
                "Those are skipped (rebuild recommended to include updates).",
                file=sys.stderr,
            )
    else:
        print(f"Found {len(pdf_paths)} PDFs. Counting chunks with {args.workers} workers...")

    work_items = [(str(p), str(docs_root), chunk_size, overlap) for p in to_process]
    total_chunks = 0
    per_pdf_chunks: List[Tuple[str, List[Tuple[int, str]]]] = []

    if work_items:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_pdf_for_chunks, wi) for wi in work_items]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Counting"):
                rel, page_chunks = fut.result()
                if page_chunks:
                    per_pdf_chunks.append((rel, page_chunks))
                    total_chunks += len(page_chunks)

    if existing_count == 0:
        print(f"Total chunks to index: {total_chunks}")
    else:
        print(f"New chunks to embed: {total_chunks}")

    if args.dry_run:
        return 0

    if existing_count == 0:
        if total_chunks == 0:
            print("No extractable text found in PDFs.", file=sys.stderr)
            return 1
        idx.init_index(max_elements=total_chunks, ef_construction=200, M=64)
        idx.set_ef(50)
    else:
        if total_chunks == 0:
            print("No new PDFs to embed. Index is up to date.")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            return 0
        idx.resize_index(existing_count + total_chunks)

    current_id = existing_count
    meta_mode = "a" if (existing_count and meta_path.exists() and not args.rebuild) else "w"
    save_every_batches = 50
    batches_since_save = 0

    with open(meta_path, meta_mode, encoding="utf-8") as meta_f:
        for rel, chunks in tqdm(per_pdf_chunks, desc="Embedding"):
            full = docs_root / rel.replace("/", os.sep)
            try:
                st = full.stat()
                sig = (int(st.st_size), int(st.st_mtime_ns))
            except Exception:
                sig = None
            for i in range(0, len(chunks), args.batch_size):
                batch = chunks[i : i + args.batch_size]
                batch_texts = [t for _, t in batch]
                vecs = embed_texts(batch_texts, base_url, embed_model, session, max_chars=embed_max_chars)
                ids = np.arange(current_id, current_id + len(batch))
                idx.add_items(vecs, ids)

                with db_conn:
                    rows = [
                        (
                            int(ids[j]),
                            rel,
                            int(i + j),
                            int(page_num),
                            chunk,
                        )
                        for j, (page_num, chunk) in enumerate(batch)
                    ]
                    db_conn.executemany(
                        "INSERT OR REPLACE INTO chunks(id,path,chunk_index,page_number,text) VALUES (?,?,?,?,?)",
                        rows,
                    )
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
                batches_since_save += 1
                if batches_since_save >= save_every_batches:
                    idx.save_index(str(index_path))
                    set_state_int(db_conn, "last_saved_id", current_id)
                    batches_since_save = 0

            if sig is not None:
                with db_conn:
                    db_conn.execute(
                        "INSERT OR REPLACE INTO files(path,size,mtime_ns,max_id) VALUES (?,?,?,?)",
                        (rel, int(sig[0]), int(sig[1]), int(current_id)),
                    )

    idx.save_index(str(index_path))
    set_state_int(db_conn, "last_saved_id", current_id)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Index written to: {index_path}")
    print(f"Metadata written to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
