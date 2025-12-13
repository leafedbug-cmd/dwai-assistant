$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Get-DockerComposeCommand {
  try {
    $null = & docker compose version 2>$null
    return @("docker", "compose")
  } catch {
    $cmd = Get-Command docker-compose -ErrorAction SilentlyContinue
    if ($cmd) { return @($cmd.Source) }
  }
  throw "Docker Compose not found. Install Docker Desktop (or docker-compose) first."
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
  # Ignore; Open WebUI can still start and will show an error until Ollama is running.
}

$compose = Get-DockerComposeCommand
Push-Location $repoRoot
try {
  & $compose up -d | Out-Host
} finally {
  Pop-Location
}

# Quick health check (best-effort).
try {
  $health = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://localhost:3000/health"
  if ($health.StatusCode -eq 200) {
    Write-Host "Open WebUI is up: http://localhost:3000"
    exit 0
  }
} catch {
  # Ignore; container may still be starting.
}

Write-Host "Open WebUI started (or starting). Open: http://localhost:3000"
