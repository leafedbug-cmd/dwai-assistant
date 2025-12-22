# DWAI Assistant

DWAI Assistant is a local document management and AI-assisted knowledge base for service documentation. It integrates Ollama (Qwen) with local RAG scripts and an optional Streamlit UI.

## Security Notes (Local-Only)

- Default operation is local-only; no documents are uploaded to external services.
- Local endpoints:
  - Ollama API: http://localhost:11434
  - Streamlit UI: http://localhost:8501
  - Open WebUI (optional Docker): http://localhost:3000
- Data at rest:
  - `docs/` holds source PDFs in their original hierarchy.
  - `data/` holds local indexes, caches, and reports and is gitignored.
- Configuration:
  - `scripts/rag_config.json` is local-only; copy from the example and do not commit secrets.
- Network egress is only required when you intentionally pull models or install dependencies.

## Local-Only Setup (no Docker/Tailscale)

If you are on the `local-only-no-docker` branch, everything runs on your Windows machine.

1. Create and activate a virtual environment, then install deps:
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r .\requirements-webui.txt
   pip install -r .\requirements-rag.txt
   ```
2. Start Ollama (and pull required models):
   ```powershell
   ollama serve
   ollama pull qwen3:4b
   ollama pull nomic-embed-text
   ollama pull llava:7b   # optional for diagrams/vision
   ```
3. Copy/edit RAG config if needed:
   ```powershell
   Copy-Item .\scripts\rag_config.example.json .\scripts\rag_config.json
   notepad .\scripts\rag_config.json
   ```
4. Build the local index (first run only):
   ```powershell
   python .\scripts\rag_reindex.py --config .\scripts\rag_config.json
   ```
5. Launch the local WebUI (Streamlit):
   ```powershell
   .\scripts\start_dwfixit_webui.ps1
   # Opens on http://localhost:8501
   ```
6. Ask questions via script or WebUI:
   ```powershell
   python .\scripts\rag_ask.py "What are the HX75 maintenance intervals?"
   # Or open http://localhost:8501 in your browser
   ```

## Repository Layout

```
dwai-assistant/
|-- README.md
|-- LICENSE
|-- docker-compose.yml
|-- .gitignore
|-- .gitattributes
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- SECURITY_REVIEW.md
|   |-- OPERATIONS.md
|   |-- setup-open-webui.md
|   `-- <service PDF folders...>
|-- scripts/
|   |-- rag_reindex.py
|   |-- rag_ask.py
|   |-- remove_non_english.py
|   |-- sync_service_docs.ps1
|   `-- start_dwfixit_webui.ps1
|-- webui/
`-- data/                      # local-only indexes, caches, reports (gitignored)
```

## Operations

For operational procedures, API usage, and troubleshooting, see `docs/OPERATIONS.md`.

## Git LFS

Large binary files (PDFs, Word docs, Excel sheets, etc.) are tracked by Git LFS:

- `*.pdf`
- `*.doc`, `*.docx`
- `*.xls`, `*.xlsx`
- `*.ppt`, `*.pptx`
- `*.csv`

Run `git lfs status` to verify LFS tracking.

## License

MIT License. See `LICENSE`.

## Review Checklist

- Confirm all services bind to localhost unless explicitly configured otherwise.
- Verify no secrets or tokens are committed (configs are example-only).
- Validate data at rest location (`docs/` for PDFs, `data/` for local indexes).
- Review outbound network requirements (model pulls and dependency installs only).
- Capture a short dependency inventory for reviewers if requested (pip freeze).
