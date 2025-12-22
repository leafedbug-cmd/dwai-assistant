param(
  [switch]$HideOllamaWindow,
  [switch]$HideWebUIWindow,
  [switch]$RestartWebUI,
  [switch]$Lan,
  [switch]$OpenFirewall
)

$ErrorActionPreference = "Stop"

function Wait-ForPort {
  param(
    [int]$Port,
    [int]$TimeoutSeconds = 30
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
      if ($listening) { return $true }
    } catch {
      # Ignore transient permission/availability issues.
    }
    Start-Sleep -Milliseconds 250
  }
  return $false
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$app = Join-Path $repoRoot "webui\\app.py"
$bindAddress = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }

if (-not (Test-Path $python)) {
  throw "Virtualenv not found at: $python. Create it first (py -3.13 -m venv .venv) and install deps."
}
if (-not (Test-Path $app)) {
  throw "WebUI app not found at: $app"
}

if ($Lan -and $OpenFirewall) {
  try {
    $rules = @(
      @{ Name = "dwFixIT WebUI (8501)"; Port = 8501 },
      @{ Name = "dwFixIT Ollama API (11434)"; Port = 11434 }
    )
    foreach ($r in $rules) {
      $existing = Get-NetFirewallRule -DisplayName $r.Name -ErrorAction SilentlyContinue
      if (-not $existing) {
        New-NetFirewallRule `
          -DisplayName $r.Name `
          -Direction Inbound `
          -Action Allow `
          -Protocol TCP `
          -LocalPort $r.Port `
          -Profile Private `
          -ErrorAction Stop | Out-Null
      }
    }
  } catch {
    Write-Warning "Failed to create firewall rules (try running PowerShell as Administrator): $($_.Exception.Message)"
  }
}

# Stop Ollama if it is running.
Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Start Ollama.
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
  throw "ollama not found on PATH. Install Ollama, then re-run this script."
}

$env:OLLAMA_HOST = "$bindAddress`:11434"

$ollamaWindowStyle = if ($HideOllamaWindow) { "Hidden" } else { "Normal" }
Start-Process -WindowStyle $ollamaWindowStyle -FilePath $ollamaCmd.Source -ArgumentList @("serve") -WorkingDirectory $repoRoot | Out-Null

if (-not (Wait-ForPort -Port 11434 -TimeoutSeconds 45)) {
  throw "Ollama did not start listening on $($env:OLLAMA_HOST) within 45s."
}

# Start Streamlit WebUI if it's not already running.
$webListening = $null
try {
  $webListening = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
} catch {
  $webListening = $null
}
if ($webListening) {
  if (-not $RestartWebUI) {
    Write-Output "WebUI already listening on http://127.0.0.1:8501"
    exit 0
  }

  $conn = $webListening | Select-Object -First 1
  $webPid = $conn.OwningProcess
  if ($webPid) {
    $proc = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $webPid) -ErrorAction SilentlyContinue
    $cmd = $proc.CommandLine
    if ($cmd -and ($cmd -like "*streamlit*") -and ($cmd -like "*app.py*")) {
      Stop-Process -Id $webPid -Force -ErrorAction SilentlyContinue
      Start-Sleep -Milliseconds 300
    } else {
      throw "Port 8501 is in use by PID $webPid. Stop it (or run without -RestartWebUI)."
    }
  }
}

$webWindowStyle = if ($HideWebUIWindow) { "Hidden" } else { "Normal" }
Start-Process `
  -WindowStyle $webWindowStyle `
  -WorkingDirectory $repoRoot `
  -FilePath $python `
  -ArgumentList @(
    "-m","streamlit","run",$app,
    "--server.address",$bindAddress,
    "--server.port","8501",
    "--server.headless","true"
  ) | Out-Null

Write-Output "Started:"
Write-Output "- Ollama: http://$bindAddress`:11434"
Write-Output "- WebUI:  http://$bindAddress`:8501"
