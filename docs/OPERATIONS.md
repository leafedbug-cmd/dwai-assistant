# DWAI Assistant Operations Guide

This document provides operational procedures, API usage examples, and troubleshooting for the DWAI Assistant system.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Syncing Service Documents](#syncing-service-documents)
4. [Container Management](#container-management)
5. [API Usage](#api-usage)
6. [Dataset Management & Reindexing](#dataset-management--reindexing)
7. [Troubleshooting](#troubleshooting)

---

## Overview

DWAI Assistant integrates:

- **Document Repository:** Git-based storage with LFS for large binaries
- **Ollama:** Local LLM inference engine with Qwen model
- **Open WebUI:** Browser-based interface for chat and document search
- **Tailscale:** Secure remote access (optional)

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Windows Host (C:\dwai-assistant)                        │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ Ollama Service   │  │ Open WebUI Container     │   │
│  │ (Port 11434)     │  │ (Port 3000 / Port 8080)  │   │
│  └──────────────────┘  │                          │   │
│         ▲              │  - Chat Interface        │   │
│         │              │  - Knowledge Base        │   │
│         │              │  - Settings              │   │
│         └─────────────→├──────────────────────────┤   │
│                        │ Mounts:                  │   │
│                        │ - /data/dwai-docs (RO)   │   │
│                        │ - /app/backend/data (RW) │   │
│                        └──────────────────────────┘   │
│                                   ▲                    │
│                                   │                    │
└───────────────────────────────────┼────────────────────┘
                                    │
                         (Tailscale or Local)
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │ Remote Device       │
                        │ Browser / API Client│
                        │ 100.80.203.54:3000  │
                        │ 100.80.203.54:11434 │
                        └─────────────────────┘
```

---

## Syncing Service Documents

### Initial Sync

Run the PowerShell sync script to copy all documents from the source to the repo:

```powershell
cd C:\Users\austin\dwai-assistant
.\scripts\sync_service_docs.ps1
```

**What it does:**
- Robocopy copies all files from `C:\Users\austin\Downloads\Service Documents\Service Documents` to `docs/service-documents`
- Preserves folder hierarchy exactly
- Only copies files newer than destination (efficient updates)
- Git LFS automatically handles large binaries

### Adding New Documents

1. **Place documents** in the appropriate category under `docs/service-documents/`
2. **Commit to Git:**
   ```powershell
   cd C:\Users\austin\dwai-assistant
   git add docs/service-documents/
   git commit -m "Add new service documents for [Category]"
   git push origin main
   ```
3. **Reindex in Open WebUI** (see [Dataset Management](#dataset-management--reindexing))

### Document Naming & Organization

**Hierarchy preserved from source:**
- AUGER BORING/
- COMPACT UTILITY EQUIPMENT/
- DIRECTIONAL DRILLS/
- ELECTRONICS/
- PARTS/
- TRENCHERPLOWSSURFACE MINERS/
- TRENCHLESS/
- VACUUM EXCAVATOR/
- [Additional categories...]

**Normalization:** If future edits are needed, document the mapping in `docs/mappings.md`.

---

## Container Management

### Check Container Status

```powershell
docker ps
docker logs open-webui  # View logs
docker stats open-webui # View resource usage
```

### Stop Container

```powershell
docker stop open-webui
```

### Restart Container

```powershell
docker restart open-webui
```

### Remove and Redeploy Container

```powershell
docker stop open-webui
docker rm open-webui
docker run -d -p 3000:8080 `
  --add-host=host.docker.internal:host-gateway `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -v C:/app/backend/data:/app/backend/data `
  -v C:/dwai-assistant/docs/service-documents:/data/dwai-docs:ro `
  --name open-webui `
  --restart always `
  ghcr.io/open-webui/open-webui:main
```

### Update Open WebUI Image

```powershell
docker pull ghcr.io/open-webui/open-webui:main
docker stop open-webui
docker rm open-webui
# Re-run container command above
```

---

## API Usage

### Ollama API: Generate Text

Generate a response from Qwen via Ollama:

```powershell
$body = @{
    model = "qwen"
    prompt = "What is the operator's manual process for HX75?"
    stream = $false
} | ConvertTo-Json

curl -X POST http://100.80.203.54:11434/api/generate `
  -ContentType "application/json" `
  -Body $body
```

**Response Example:**
```json
{
  "model": "qwen",
  "created_at": "2025-12-09T12:34:56Z",
  "response": "The operator's manual for the HX75 includes... [response text]",
  "done": true,
  "context": [200, 400, ...],
  "total_duration": 3500000000,
  "load_duration": 500000000,
  "prompt_eval_count": 15,
  "prompt_eval_duration": 800000000,
  "eval_count": 120,
  "eval_duration": 2200000000
}
```

### Ollama API: List Available Models

```powershell
curl http://100.80.203.54:11434/api/tags
```

### Open WebUI: Test Chat Interface

1. Visit `http://100.80.203.54:3000` (or `http://localhost:3000`)
2. Ensure Ollama Base URL is set: **Settings → Connections → http://host.docker.internal:11434**
3. Type a query and press Send

---

## Dataset Management & Reindexing

### Manual Reindex via UI

1. **Settings → Datasets**
2. Find "DWAI Service Documents"
3. Click **Reindex** or **Update**
4. Monitor indexing progress

### Automated Reindexing Script (Optional)

**File:** `scripts/reindex_openwebui.ps1`

```powershell
# Example: Trigger reindex via Open WebUI API
# (Requires API token; see Open WebUI settings)

$apiUrl = "http://100.80.203.54:3000/api"
$token = "<YOUR_API_TOKEN>"

$headers = @{
    Authorization = "Bearer $token"
    "Content-Type" = "application/json"
}

$body = @{
    name = "DWAI Service Documents"
} | ConvertTo-Json

Invoke-RestMethod -Uri "$apiUrl/datasets/reindex" `
  -Method POST `
  -Headers $headers `
  -Body $body
```

### Check Index Status

In Open WebUI UI:
1. **Settings → Datasets**
2. Hover over "DWAI Service Documents" to see:
   - Total indexed files
   - Last updated timestamp
   - File types and sizes

---

## Troubleshooting

### Cannot connect to Open WebUI

**Symptom:** Browser shows connection error

**Solutions:**
- Verify container is running: `docker ps | grep open-webui`
- Check logs: `docker logs open-webui`
- Verify port 3000 is not blocked by firewall
- Test local connection first: `http://localhost:3000`
- For Tailscale: Confirm device is online and VPN connection is active

### Ollama not responding to queries

**Symptom:** Open WebUI shows "Model error" or timeout

**Solutions:**
- Verify Ollama is running: `ollama list`
- Check Qwen model exists: `ollama list | findstr qwen`
- Test Ollama directly: `curl http://localhost:11434/api/tags`
- Verify container's `OLLAMA_BASE_URL` environment variable: `docker inspect open-webui | findstr OLLAMA`
- Restart Ollama if necessary

### Documents not found in search

**Symptom:** Knowledge base is empty or queries don't reference documents

**Solutions:**
- Verify mount path exists: `docker exec open-webui ls -la /data/dwai-docs`
- Check file permissions (should be readable): `docker exec open-webui find /data/dwai-docs -type f | head -5`
- Confirm dataset is created and reindexed: **Settings → Datasets**
- Check file types are supported (PDF, DOCX, TXT, CSV)
- Look for indexing errors in `docker logs open-webui`

### Disk space issues

**Symptom:** Docker reports disk space error or slow indexing

**Solutions:**
- Check Docker disk usage: `docker system df`
- Prune old images/containers: `docker system prune -a`
- Check host disk: `Get-Volume`
- Verify LFS files are tracked correctly: `git lfs ls-files | wc -l`

### Git LFS issues

**Symptom:** Large files not syncing or showing as "LFS pointers"

**Solutions:**
- Verify Git LFS is installed: `git lfs version`
- Check `.gitattributes` is committed: `git show HEAD:.gitattributes`
- Pull LFS files: `git lfs pull`
- Monitor LFS status: `git lfs status`

### Sync script errors

**Symptom:** `sync_service_docs.ps1` fails or hangs

**Solutions:**
- Check source path exists: `Test-Path "C:\Users\austin\Downloads\Service Documents\Service Documents"`
- Ensure robocopy is available: `robocopy /?`
- Run with verbose output: Edit script and add `/V` flag to robocopy
- Check antivirus isn't blocking file operations
- Verify disk space on destination

---

## Common Tasks

### Query Qwen about a specific document

```powershell
# In Open WebUI, use chat interface:
# "Based on the service documents, what are the maintenance intervals for HX75?"
```

### Update service documents and re-index

```powershell
cd C:\Users\austin\dwai-assistant

# Sync new files
.\scripts\sync_service_docs.ps1

# Commit
git add docs/service-documents/
git commit -m "Update service documents"
git push

# Reindex in Open WebUI UI: Settings → Datasets → DWAI Service Documents → Reindex
```

### Export chat history

1. Open WebUI UI → Chat → Export (if available)
2. Or query Open WebUI API for chat records

### Backup data

```powershell
# Backup Open WebUI data volume
docker run --rm -v C:/app/backend/data:/data -v C:/backups:/backup `
  alpine tar czf /backup/open-webui-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss').tar.gz -C /data .
```

---

## Performance Tuning

### Increase Ollama context window

Modify Ollama configuration (on host):
```bash
ollama run qwen --num_ctx 4096
```

### Optimize Open WebUI container resources

```powershell
docker update --cpus=2 --memory=4g open-webui
docker restart open-webui
```

### Monitor performance

```powershell
docker stats open-webui --no-stream
```

---

## Support & References

- **Open WebUI Issues:** https://github.com/open-webui/open-webui/issues
- **Ollama Documentation:** https://ollama.ai/
- **Docker Documentation:** https://docs.docker.com/
- **Qwen Model:** https://huggingface.co/Qwen/Qwen-7B

