# DWAI Assistant

A document management and AI-assisted repository for DeWALT service documentation integrated with Ollama/Qwen and Open WebUI.

## Local-Only Branch (no Docker/Tailscale)

You are on the `local-only-no-docker` branch. This branch is meant for strictly local use and does not rely on Docker or Tailscale. Everything runs on your Windows machine:

- Ollama serves models on `localhost:11434`
- The included Streamlit UI serves on `localhost:8501`
- RAG indexing and queries use local scripts in `scripts/`

### Quick Local Setup

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

## Overview

This repository houses the complete DeWALT service documentation hierarchy and serves as a knowledge source for the DWAI Assistant system. Documents are indexed by Open WebUI and made searchable through Qwen, a powerful language model running on Ollama.

## Repository Structure

```
dwai-assistant/
├── README.md                          # This file
├── docker-compose.yml                 # Open WebUI (Docker) launcher
├── .gitignore                         # Git ignore patterns
├── .gitattributes                     # Git LFS configuration
├── LICENSE                            # Repository license
├── docs/                              # Service documents (PDF hierarchy)
│   ├── AUGER BORING/
│   ├── COMPACT UTILITY EQUIPMENT/
│   ├── DIRECTIONAL DRILLS/
│   ├── ELECTRONICS/
│   ├── PARTS/
│   ├── TRENCHERPLOWSSURFACE MINERS/
│   ├── TRENCHLESS/
│   ├── VACUUM EXCAVATOR/
│   ├── setup-open-webui.md            # Setup instructions for Open WebUI/Ollama integration
│   └── OPERATIONS.md                  # Operational procedures and API documentation
├── scripts/
│   ├── rag_reindex.py                 # Build local RAG index (fast, no upload)
│   ├── rag_ask.py                     # Query local RAG index
│   ├── sync_service_docs.ps1          # PowerShell script to sync docs from source
│   └── reindex_openwebui.ps1          # (Optional) Helper to reindex Open WebUI dataset
└── webui/                             # Simple local Streamlit UI (optional)
```

## Quick Start

### Prerequisites

- Git and Git LFS installed
- Ollama running with Qwen model
- Local network access (optional; this branch targets localhost)

### Local Setup

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/leafedbug-cmd/dwai-assistant.git
   cd dwai-assistant
   ```

2. **Sync service documents:**
   ```powershell
   .\scripts\sync_service_docs.ps1
   ```

3. **Stage and commit changes:**
   ```powershell
   git add .
   git commit -m "Initial service documents import"
   git push origin main
   ```

## Local RAG (No Upload Needed)

Open WebUI in your current build only supports small “upload” knowledge bases. For your 30GB+ PDF library, use the local RAG scripts in `scripts/`.

1. Install Python 3.10+ and dependencies:
   ```powershell
   pip install -r .\requirements-rag.txt
   ```
   Make sure your embedding model is available in Ollama:
   ```powershell
   ollama pull nomic-embed-text
   ```
2. (Optional) copy and edit config:
   ```powershell
   Copy-Item .\scripts\rag_config.example.json .\scripts\rag_config.json
   notepad .\scripts\rag_config.json
   ```
3. Build the index (first time takes a while):
   ```powershell
   python .\scripts\rag_reindex.py --config .\scripts\rag_config.json
   ```
4. Ask questions:
   ```powershell
   python .\scripts\rag_ask.py "What are the HX75 maintenance intervals?"
   ```

### Diagram / Vision Mode (optional)

For manuals where answers depend on exploded diagrams and callout numbers, you can run a local vision model on the top retrieved pages:

1. Pull a vision model that Ollama supports (example):
   ```powershell
   ollama pull llava:7b
   ```
2. Rebuild the index if you haven’t since updating scripts (stores page numbers):
   ```powershell
   python .\scripts\rag_reindex.py --config .\scripts\rag_config.json
   ```
3. Vision auto-triggers for diagram/callout questions. You can still force or disable it:
   ```powershell
   python .\scripts\rag_ask.py "What is callout 12 on the JT20 exploded view?"
   python .\scripts\rag_ask.py --vision "Force vision even if not needed"
   python .\scripts\rag_ask.py --no-vision "Disable vision for this query"
   ```

### Docker Container Setup

Docker/Tailscale are not required on this branch. If you prefer Docker, see [docs/setup-open-webui.md](docs/setup-open-webui.md) (optional).

## Connection Endpoints

- **dwFixIT WebUI (Streamlit):** http://localhost:8501
- **Open WebUI (optional, Docker):** http://localhost:3000
- **Ollama API:** http://localhost:11434

Note: For LAN access from another PC, start Ollama bound to all interfaces by setting `OLLAMA_HOST=0.0.0.0` before `ollama serve`, then use `http://<your-host-ip>:11434`.

## Operations

For detailed operational procedures, API usage, and troubleshooting, see [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Document Organization

All service documents are preserved in their original folder hierarchy. Categories include:

- AUGER BORING
- COMPACT UTILITY EQUIPMENT
- DIRECTIONAL DRILLS
- ELECTRONICS
- PARTS
- TRENCHERPLOWSSURFACE MINERS
- TRENCHLESS
- VACUUM EXCAVATOR

**Important:** Do not rename or flatten directories. The hierarchy is maintained for context-rich retrieval by the AI model.

## Git LFS

Large binary files (PDFs, Word docs, Excel sheets, etc.) are tracked by Git LFS:

- `*.pdf`
- `*.doc`, `*.docx`
- `*.xls`, `*.xlsx`
- `*.ppt`, `*.pptx`
- `*.csv`

Run `git lfs status` to verify LFS tracking.

## License

[Choose appropriate license - default: MIT]

## Contacts & Support

For questions or issues, please open a GitHub issue or contact the repository maintainer.
