# Security Review Notes

This document summarizes the security posture of the DWAI Assistant local-only workflow.

## Scope

- Local Windows workstation deployment.
- Local RAG scripts and optional Streamlit UI.
- Optional Docker-based Open WebUI (not required on the local-only branch).

## Data Handling

- Source PDFs live in `docs/` and are never uploaded by default.
- Local indexes, caches, and reports live in `data/` (gitignored).
- Config files are local-only; `scripts/rag_config.json` is copied from an example.

## Network and Ports

Local endpoints:

- Ollama API: http://localhost:11434
- Streamlit UI: http://localhost:8501
- Open WebUI (optional Docker): http://localhost:3000

Outbound network is only required when you intentionally pull models (`ollama pull`) or install dependencies (`pip install`).

## Secrets and Credentials

- No secrets are committed to this repository.
- Example tokens in docs use placeholders only.
- Keep local config files out of git (`scripts/rag_config.json`).

## Access Control

- Access is limited to the local workstation user.
- Binding to `0.0.0.0` is optional and should be controlled by firewall rules.

## Logging and Retention

- Reports and temporary outputs should be stored under `data/` and reviewed before sharing.
- No telemetry is sent by default.

## Reviewer Checklist

- Confirm services bind to localhost unless explicitly configured otherwise.
- Verify no secrets are present in the repo history or working tree.
- Validate local-only data paths (`docs/`, `data/`).
- Review any optional Docker/Open WebUI usage and exposed ports.
