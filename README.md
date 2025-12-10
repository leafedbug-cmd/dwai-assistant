# DWAI Assistant

A document management and AI-assisted repository for DeWALT service documentation integrated with Ollama/Qwen and Open WebUI.

## Overview

This repository houses the complete DeWALT service documentation hierarchy and serves as a knowledge source for the DWAI Assistant system. Documents are indexed by Open WebUI and made searchable through Qwen, a powerful language model running on Ollama.

## Repository Structure

```
dwai-assistant/
├── README.md                          # This file
├── .gitignore                         # Git ignore patterns
├── .gitattributes                     # Git LFS configuration
├── LICENSE                            # Repository license
├── docs/
│   ├── service-documents/             # Mirror of C:\Users\austin\Downloads\Service Documents
│   │   ├── AUGER BORING/
│   │   ├── COMPACT UTILITY EQUIPMENT/
│   │   ├── DIRECTIONAL DRILLS/
│   │   ├── ELECTRONICS/
│   │   ├── PARTS/
│   │   ├── TRENCHERPLOWSSURFACE MINERS/
│   │   ├── TRENCHLESS/
│   │   ├── VACUUM EXCAVATOR/
│   │   └── [additional categories...]
│   ├── setup-open-webui.md            # Setup instructions for Open WebUI/Ollama integration
│   ├── OPERATIONS.md                  # Operational procedures and API documentation
│   ├── index.csv                      # (Auto-generated) File listing with paths and metadata
│   └── mappings.md                    # (If needed) Document name mappings and normalization log
└── scripts/
    ├── sync_service_docs.ps1          # PowerShell script to sync docs from source
    └── reindex_openwebui.ps1          # (Optional) Helper to reindex Open WebUI dataset
```

## Quick Start

### Prerequisites

- Git and Git LFS installed
- Docker with Open WebUI container deployed
- Ollama running with Qwen model
- Tailscale (for remote access) or local network access

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

### Docker Container Setup

See [docs/setup-open-webui.md](docs/setup-open-webui.md) for detailed instructions on running and configuring the Open WebUI container.

## Connection Endpoints

- **Open WebUI:** `http://100.80.203.54:3000` (Tailscale) or `http://localhost:3000` (local)
- **Ollama API:** `http://100.80.203.54:11434` (Tailscale) or `http://localhost:11434` (local)
- **Ollama Host:** `0.0.0.0`

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
