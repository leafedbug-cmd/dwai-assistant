# Architecture Overview

DWAI Assistant runs entirely on a local Windows workstation in the default setup. It uses Ollama for model serving, local RAG scripts for indexing and retrieval, and an optional Streamlit UI.

## Components

- Ollama: local model server on `localhost:11434`.
- RAG scripts: `scripts/rag_reindex.py` and `scripts/rag_ask.py`.
- Streamlit UI: `scripts/start_dwfixit_webui.ps1` serving on `localhost:8501`.
- Optional Docker Open WebUI: `docker-compose.yml` (not required for local-only).

## Data Flow Diagram

```
--------+        +------------------+        +--------------------+
|  User | <----> | Streamlit WebUI  | <----> |  rag_ask.py (RAG)   |
--------+        +------------------+        +--------------------+
                                               |           |
                                               |           +--> Vector index (data/rag)
                                               |           +--> Source PDFs (docs/)
                                               |
                                               +--> Ollama API (localhost:11434)

Optional path:
  User <----> Open WebUI (Docker) <----> Ollama API (localhost:11434)
```

## Storage Locations

- `docs/`: source PDFs in their original hierarchy.
- `data/`: local-only indexes, caches, and reports (gitignored).
- `.venv/`: local Python environment.

## Ports

- 11434: Ollama API
- 8501: Streamlit UI
- 3000: Open WebUI (optional)
