$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$app = Join-Path $repoRoot "webui\\app.py"

if (-not (Test-Path $python)) {
  throw "Virtualenv not found at: $python. Create it first (py -3.13 -m venv .venv) and install deps."
}
if (-not (Test-Path $app)) {
  throw "WebUI app not found at: $app"
}

# Avoid starting a second server if it's already listening.
try {
  $listening = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
  if ($listening) { exit 0 }
} catch {
  # Ignore; continue.
}

# Best-effort: start Ollama if available and not running.
try {
  $ollamaRunning = Get-Process ollama -ErrorAction SilentlyContinue
  if (-not $ollamaRunning) {
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaCmd) {
      Start-Process -WindowStyle Hidden -FilePath $ollamaCmd.Source -ArgumentList "serve" -WorkingDirectory $repoRoot | Out-Null
    }
  }
} catch {
  # Ignore; WebUI can still start and will error on first request if Ollama isn't running.
}

# Start Streamlit in headless mode and bind to localhost.
Start-Process `
  -WindowStyle Hidden `
  -WorkingDirectory $repoRoot `
  -FilePath $python `
  -ArgumentList @(
    "-m","streamlit","run",$app,
    "--server.address","127.0.0.1",
    "--server.port","8501",
    "--server.headless","true"
  ) | Out-Null
