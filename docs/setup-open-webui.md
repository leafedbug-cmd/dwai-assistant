# Open WebUI Setup & Ollama Integration

This document provides step-by-step instructions for running the Open WebUI container, configuring it with Ollama, and connecting the DWAI service documents.

> Note (local-only branch): Docker and Tailscale are not required on this branch. Use this document only if you choose to run Open WebUI in Docker; otherwise, start the local Streamlit UI with `scripts/start_dwfixit_webui.ps1`.

## Prerequisites

- **Docker** installed and running
- **Ollama** running on the Windows host with Qwen model pulled
- **Repository** cloned to `C:\Users\austin\dwai-assistant`
- **Tailscale** (optional for remote access; not used in local-only branch) or local network connectivity

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

### Option A: Docker Compose (recommended)

Create `docker-compose.yml` in the repo root (already added):

```yaml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    volumes:
      - open-webui-data:/app/backend/data
      - ./docs:/data/dwai-docs:ro
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped

volumes:
  open-webui-data:
```

Start/stop:

```powershell
docker compose up -d
docker compose down
```

If you cloned this repo, you can also use the helper:

```powershell
.\scripts\start_open_webui.ps1
```

### Verify connectivity to host Ollama and docs (optional but recommended)

```powershell
# From Windows host, confirm the container is healthy
curl -s http://localhost:3000/health

# From inside the container, confirm Ollama and the docs mount are reachable
docker exec open-webui sh -c "curl -s http://host.docker.internal:11434/api/tags | head"
docker exec open-webui sh -c "ls /data/dwai-docs | head"
```

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
  -v C:/path/to/dwai-assistant/docs:/data/dwai-docs:ro `
  --name open-webui `
  --restart always `
  ghcr.io/open-webui/open-webui:main
```

**Notes:**
- `--add-host=host.docker.internal:host-gateway` allows the container to reach the Windows host.
- `-e OLLAMA_BASE_URL=http://host.docker.internal:11434` points to Ollama running on the host.
- `-v C:/app/backend/data:/app/backend/data` persists Open WebUI data.
- `-v C:/path/to/dwai-assistant/docs:/data/dwai-docs:ro` mounts docs read-only for indexing.

## Step 3: Access Open WebUI

Once running, access the web interface:

- **Local:** `http://localhost:3000`
- **Tailscale:** `http://100.80.203.54:3000`

## Step 4: Configure Open WebUI Settings

1. **Login** with default credentials (if first-time setup).
2. **Settings → Connections:**
   - Ollama Base URL: `http://host.docker.internal:11434`
   - Default Model: `qwen3:8b` (or your deployed variant)
3. **Test Connection** to ensure Ollama is reachable.

## Step 5: Index Service Documents (Knowledge Base)

### Create a New Dataset/Knowledge Base

1. Navigate to **Settings → Datasets** (or **Knowledge** in some versions).
2. Click **Create New Dataset/Knowledge Base**.
3. **Name:** `DWAI Service Documents`
4. **Source:** Select folder path input.
5. **Path:** `/data/dwai-docs` (inside container; maps to `C:\Users\austin\dwai-assistant\docs\service-documents` on host).
   - If you used the `docker-compose.yml` in this repo, it maps to your repo's `.\docs` folder.
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
  -v C:/path/to/dwai-assistant/docs:/data/dwai-docs:ro `
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

## Feedback Loop (Local RAG)

If you use the included `webui/` (Streamlit) UI, you can save feedback on whether an answer was correct.
Feedback is written to `data/feedback/feedback.jsonl` and is used to speed up future parts lookups when the same
model + short part description repeats.

## Additional Resources

- [Open WebUI Documentation](https://openwebui.com/)
- [Ollama Documentation](https://ollama.ai/)
- [Qwen Model Card](https://huggingface.co/Qwen)
