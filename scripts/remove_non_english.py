import argparse
import os
import re
import shutil
import csv
from pathlib import Path

import fitz  # PyMuPDF


LANG_TOKENS = [
    "spanish",
    "espanol",
    "español",
    "castellano",
    "french",
    "francais",
    "français",
    "german",
    "deutsch",
    "italian",
    "italiano",
    "portuguese",
    "portugues",
    "português",
    "brazil",
    "brasil",
    "dutch",
    "nederlands",
    "swedish",
    "svenska",
    "norwegian",
    "norsk",
    "danish",
    "dansk",
    "finnish",
    "suomi",
    "polish",
    "polski",
    "czech",
    "cesky",
    "česky",
    "slovak",
    "slovensky",
    "hungarian",
    "magyar",
    "romanian",
    "română",
    "romana",
    "bulgarian",
    "български",
    "greek",
    "ελληνικά",
    "russian",
    "русский",
    "ukrainian",
    "українська",
    "turkish",
    "türkçe",
    "arabic",
    "العربية",
    "hebrew",
    "עברית",
    "chinese",
    "中文",
    "mandarin",
    "japanese",
    "日本語",
    "korean",
    "한국어",
    "thai",
    "ไทย",
    "vietnamese",
    "tiếng việt",
]


STOPWORDS = {
    "en": {
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "with",
        "on",
        "is",
        "are",
        "this",
        "that",
        "be",
        "as",
        "from",
        "by",
        "not",
        "do",
        "if",
        "you",
        "your",
    },
    # Keep these biased toward >=3-char tokens to avoid false positives on schematics/abbreviations.
    "es": {"para", "con", "por", "este", "esta", "estos", "estas", "su", "sus", "manual", "operador", "seguridad"},
    "fr": {"pour", "avec", "sans", "votre", "cette", "manuel", "opérateur", "sécurité"},
    "de": {"nicht", "und", "oder", "für", "mit", "sind", "dies", "diese", "handbuch", "bediener", "sicherheit"},
    "it": {"non", "per", "con", "questo", "questa", "manuale", "operatore", "sicurezza"},
    "pt": {"para", "com", "por", "não", "este", "esta", "manual", "operador", "segurança"},
}


RE_WORD = re.compile(r"[a-zA-ZÀ-ÿ]+")
RE_NON_LATIN = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]"
)

# Suppress noisy MuPDF parse warnings from some PDFs.
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass


def looks_non_english_by_path(rel: str) -> bool:
    low = rel.lower()
    for t in LANG_TOKENS:
        if t in low:
            return True
    return False


def extract_text_sample(pdf_path: Path, max_pages: int, max_chars: int) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    out = []
    try:
        pages = min(max_pages, doc.page_count)
        for i in range(pages):
            try:
                txt = doc.load_page(i).get_text("text") or ""
            except Exception:
                txt = ""
            if txt:
                out.append(txt)
            if sum(len(s) for s in out) >= max_chars:
                break
    finally:
        doc.close()
    text = "\n".join(out)
    return text[:max_chars]

def try_langid(text: str) -> tuple[str, float] | None:
    """
    Optional language ID (more robust than stopword heuristics).
    Returns (lang, confidence) or None if langid isn't installed.
    """
    try:
        import langid  # type: ignore
    except Exception:
        return None
    try:
        lang, score = langid.classify(text)
        # langid score is log-prob-ish; normalize to a rough 0..1 using a sigmoid.
        conf = 1.0 / (1.0 + (2.718281828 ** (-score)))
        return lang, float(conf)
    except Exception:
        return None


def score_language(text: str, min_word_len: int) -> tuple[str, dict[str, int], int]:
    words = [w.lower() for w in RE_WORD.findall(text) if len(w) >= min_word_len]
    if not words:
        return "unknown", {k: 0 for k in STOPWORDS}, 0
    counts = {k: 0 for k in STOPWORDS}
    for w in words:
        for lang, sw in STOPWORDS.items():
            if w in sw:
                counts[lang] += 1
    best_lang = max(counts, key=lambda k: counts[k])
    if counts[best_lang] == 0:
        return "unknown", counts, len(words)
    return best_lang, counts, len(words)


def is_confident_non_english(
    text: str,
    min_hits: int,
    ratio: float,
    min_word_len: int,
    min_total_words: int,
    min_distinct_non_en_stopwords: int,
) -> tuple[bool, str, dict[str, int], int, str]:
    if RE_NON_LATIN.search(text):
        return True, "non_latin", {k: 0 for k in STOPWORDS}, 0, "non_latin_chars"

    lang, counts, total_words = score_language(text, min_word_len=min_word_len)
    if lang in ("unknown", "en"):
        return False, lang, counts, total_words, "unknown_or_en"
    best = counts.get(lang, 0)
    en = counts.get("en", 0)
    if best < min_hits:
        return False, lang, counts, total_words, "min_hits"
    if total_words < min_total_words:
        return False, lang, counts, total_words, "min_total_words"

    # Distinct stopword guard: avoid misclassifying schematics with many short abbreviations.
    distinct = 0
    for w in {w.lower() for w in RE_WORD.findall(text) if len(w) >= min_word_len}:
        if w in STOPWORDS.get(lang, set()):
            distinct += 1
    if distinct < min_distinct_non_en_stopwords:
        return False, lang, counts, total_words, "min_distinct"

    if en == 0:
        return True, lang, counts, total_words, "no_en_hits"
    ok = (best / max(1, en)) >= ratio
    return ok, lang, counts, total_words, "ratio" if ok else "ratio_fail"


def looks_englishish(text: str) -> bool:
    """
    Quick keep-guard: many technical PDFs are sparse; don't delete based on weak evidence.
    """
    t = (text or "").lower()
    if not t:
        return False
    english_markers = [
        "warning",
        "caution",
        "danger",
        "important",
        "operator",
        "service",
        "maintenance",
        "troubleshooting",
        "procedure",
        "note:",
    ]
    return any(m in t for m in english_markers)


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove or quarantine non-English files under docs/.")
    ap.add_argument("--docs-root", default="docs", help="Docs root folder.")
    ap.add_argument("--mode", choices=["move", "delete"], default="move", help="move=quarantine, delete=permanent delete.")
    ap.add_argument("--quarantine-dir", default="docs/_non_english_quarantine", help="Where to move files in move mode.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen; do not modify files.")
    ap.add_argument("--max-pages", type=int, default=2, help="PDF pages to sample for language detection.")
    ap.add_argument("--max-chars", type=int, default=6000, help="Max characters to sample from a PDF.")
    ap.add_argument("--min-hits", type=int, default=15, help="Minimum stopword hits before classifying non-English.")
    ap.add_argument("--ratio", type=float, default=2.0, help="Non-English/en stopword ratio threshold.")
    ap.add_argument("--min-word-len", type=int, default=3, help="Minimum word length to count for stopword scoring.")
    ap.add_argument("--min-total-words", type=int, default=200, help="Minimum sampled words to classify a PDF by language.")
    ap.add_argument(
        "--min-distinct",
        type=int,
        default=6,
        help="Minimum distinct non-English stopwords to classify a PDF by language.",
    )
    ap.add_argument(
        "--langid-min-conf",
        type=float,
        default=0.85,
        help="If langid is installed, confidence required to treat a PDF as non-English.",
    )
    ap.add_argument(
        "--report-csv",
        default="",
        help="Optional CSV path to write decisions (relative to CWD is fine).",
    )
    args = ap.parse_args()

    docs_root = Path(args.docs_root).resolve()
    if not docs_root.exists():
        raise SystemExit(f"Docs root not found: {docs_root}")

    quarantine_dir = Path(args.quarantine_dir).resolve()
    if args.mode == "move" and not args.dry_run:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    removed = 0
    moved = 0
    scanned = 0
    flagged_path = 0
    flagged_pdf = 0
    kept = 0
    report_rows: list[dict] = []

    for p in docs_root.rglob("*"):
        if not p.is_file():
            continue
        scanned += 1
        rel = str(p.relative_to(docs_root)).replace("\\", "/")
        by_path = looks_non_english_by_path(rel)

        by_pdf = False
        pdf_reason = ""
        pdf_lang = ""
        pdf_counts: dict[str, int] | None = None
        pdf_words = 0
        langid_lang = ""
        langid_conf = 0.0
        if p.suffix.lower() == ".pdf" and not by_path:
            sample = extract_text_sample(p, max_pages=args.max_pages, max_chars=args.max_chars)
            if looks_englishish(sample):
                by_pdf = False
                pdf_reason = "english_markers"
            else:
                # Prefer langid if available.
                li = try_langid(sample)
                if li is not None:
                    langid_lang, langid_conf = li
                    if langid_lang and langid_lang != "en" and langid_conf >= args.langid_min_conf:
                        by_pdf = True
                        pdf_reason = f"langid({langid_lang},{langid_conf:.2f})"
                if not by_pdf:
                    by_pdf, pdf_lang, pdf_counts, pdf_words, pdf_reason = is_confident_non_english(
                        sample,
                        min_hits=args.min_hits,
                        ratio=args.ratio,
                        min_word_len=args.min_word_len,
                        min_total_words=args.min_total_words,
                        min_distinct_non_en_stopwords=args.min_distinct,
                    )

        if not (by_path or by_pdf):
            kept += 1
            report_rows.append(
                {
                    "relative_path": rel,
                    "decision": "keep",
                    "reason": "not_flagged",
                    "ext": p.suffix.lower(),
                }
            )
            continue

        if by_path:
            flagged_path += 1
        if by_pdf:
            flagged_pdf += 1

        reason_bits = []
        if by_path:
            reason_bits.append("path_token")
        if by_pdf:
            if pdf_reason == "non_latin_chars":
                reason_bits.append("pdf_non_latin")
            else:
                if pdf_reason.startswith("langid("):
                    reason_bits.append(pdf_reason)
                else:
                    reason_bits.append(f"pdf_lang={pdf_lang}({pdf_reason},words={pdf_words})")

        reason = ",".join(reason_bits) if reason_bits else "unknown"

        if args.mode == "delete":
            action = "DELETE"
            if not args.dry_run:
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    # Best-effort: skip failures
                    pass
        else:
            action = "MOVE"
            dest = quarantine_dir / rel
            if not args.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(p), str(dest))
                    moved += 1
                except Exception:
                    pass

        print(f"{action}: {rel} [{reason}]")
        report_rows.append(
            {
                "relative_path": rel,
                "decision": "delete" if args.mode == "delete" else "move",
                "reason": reason,
                "ext": p.suffix.lower(),
                "langid_lang": langid_lang,
                "langid_conf": f"{langid_conf:.2f}" if langid_lang else "",
            }
        )

    print("")
    print(f"Docs root: {docs_root}")
    print(f"Mode: {args.mode} (dry_run={args.dry_run})")
    if args.mode == "move":
        print(f"Quarantine: {quarantine_dir}")
    print(f"Files scanned: {scanned}")
    print(f"Flagged by path tokens: {flagged_path}")
    print(f"Flagged by PDF language: {flagged_pdf}")
    print(f"Kept: {kept}")
    print(f"Moved: {moved}")
    print(f"Deleted: {removed}")

    if args.report_csv:
        report_path = Path(args.report_csv).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for r in report_rows for k in r.keys()})
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in report_rows:
                w.writerow(r)
        print(f"Report CSV: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
