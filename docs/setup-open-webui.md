# Open WebUI Setup & Ollama Integration

This document provides step-by-step instructions for running the Open WebUI container, configuring it with Ollama, and connecting the DWAI service documents.

## Prerequisites

- **Docker** installed and running
- **Ollama** running on the Windows host with Qwen model pulled
- **Repository** cloned to `C:\dwai-assistant`
- **Tailscale** (for remote access) or local network connectivity

## Step 1: Verify Ollama is Running

### Check Ollama API

```powershell
# Test connectivity to Ollama API
curl http://localhost:11434/api/generate -Method POST -Body @{
    model = "qwen"
    prompt = "Hello"
    stream = $false
} | ConvertFrom-Json
```

### Pull Qwen Model (if not already present)

```bash
ollama pull qwen
```

## Step 2: Run Open WebUI Container

### Basic Docker Run Command (without repo mount)

```powershell
docker run -d -p 3000:8080 `
  --add-host=host.docker.internal:host-gateway `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -v C:/app/backend/data:/app/backend/data `
  --name open-webui `
  --restart always `
  ghcr.io/open-webui/open-webui:main
```

### Enhanced Command (with DWAI docs mount)

For AI-assisted document retrieval, add the service documents mount:

```powershell
docker run -d -p 3000:8080 `
  --add-host=host.docker.internal:host-gateway `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -v C:/app/backend/data:/app/backend/data `
  -v C:/dwai-assistant/docs/service-documents:/data/dwai-docs:ro `
  --name open-webui `
  --restart always `
  ghcr.io/open-webui/open-webui:main
```

**Notes:**
- `--add-host=host.docker.internal:host-gateway` allows the container to reach the Windows host.
- `-e OLLAMA_BASE_URL=http://host.docker.internal:11434` points to Ollama running on the host.
- `-v C:/app/backend/data:/app/backend/data` persists Open WebUI data.
- `-v C:/dwai-assistant/docs/service-documents:/data/dwai-docs:ro` mounts docs read-only for indexing.

## Step 3: Access Open WebUI

Once running, access the web interface:

- **Local:** `http://localhost:3000`
- **Tailscale:** `http://100.80.203.54:3000`

## Step 4: Configure Open WebUI Settings

1. **Login** with default credentials (if first-time setup).
2. **Settings → Connections:**
   - Ollama Base URL: `http://host.docker.internal:11434`
   - Default Model: `qwen` (or your deployed variant)
3. **Test Connection** to ensure Ollama is reachable.

## Step 5: Index Service Documents (Knowledge Base)

### Create a New Dataset/Knowledge Base

1. Navigate to **Settings → Datasets** (or **Knowledge** in some versions).
2. Click **Create New Dataset/Knowledge Base**.
3. **Name:** `DWAI Service Documents`
4. **Source:** Select folder path input.
5. **Path:** `/data/dwai-docs` (inside container; maps to `C:\dwai-assistant\docs\service-documents` on host).
6. **Settings:**
   - Keep directory hierarchy intact.
   - Enable recursive indexing.
   - Confirm file type support (PDF, DOCX, etc.).
7. **Start Indexing** and wait for completion (may take several minutes depending on document volume).

### Verify Indexing

- Check **Datasets** page to see indexed file count and status.
- Test with a sample query: *"Find the operator's manual for HX75"*
- Verify that Qwen cites the document path in its response.

## Step 6: Update or Restart Container

### Stop and Remove Old Container

```powershell
docker stop open-webui
docker rm open-webui
```

### Run Updated Container (with Service Docs)

```powershell
docker run -d -p 3000:8080 `
  --add-host=host.docker.internal:host-gateway `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -v C:/app/backend/data:/app/backend/data `
  -v C:/dwai-assistant/docs/service-documents:/data/dwai-docs:ro `
  --name open-webui `
  --restart always `
  ghcr.io/open-webui/open-webui:main
```

## Troubleshooting

### Container won't start

```powershell
docker logs open-webui
```

### Can't reach Ollama from container

- Verify Ollama is running on the host: `ollama list`
- Check firewall allows localhost:11434
- Ensure `--add-host=host.docker.internal:host-gateway` is in the run command

### Documents not indexing

- Confirm `/data/dwai-docs` path is mounted and readable inside container
- Check Open WebUI logs: `docker logs open-webui | grep -i index`
- Ensure documents are in supported formats (PDF, DOCX, TXT, CSV, etc.)

### Qwen model not responding

- Test Ollama directly: `curl http://localhost:11434/api/generate` with a simple prompt
- Verify `OLLAMA_BASE_URL` environment variable is set correctly in container
- Check if Qwen model is fully pulled: `ollama list`

## Next Steps

- See [docs/OPERATIONS.md](OPERATIONS.md) for operational procedures and API usage.
- Sync additional documents: `.\scripts\sync_service_docs.ps1`
- (Optional) Set up automated reindexing with `scripts/reindex_openwebui.ps1`

## Additional Resources

- [Open WebUI Documentation](https://openwebui.com/)
- [Ollama Documentation](https://ollama.ai/)
- [Qwen Model Card](https://huggingface.co/Qwen)
