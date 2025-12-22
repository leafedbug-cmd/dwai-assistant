param(
  [string]$OutDir = "",
  [switch]$IncludeDocs,
  [switch]$IncludeIndex,
  [switch]$IncludeFeedback,
  [switch]$Zip = $true
)

$ErrorActionPreference = "Stop"

function Resolve-DesktopPath {
  $desktop = Join-Path $env:USERPROFILE "Desktop"
  if (Test-Path $desktop) { return $desktop }

  $oneDriveDesktop = Join-Path $env:USERPROFILE "OneDrive\Desktop"
  if (Test-Path $oneDriveDesktop) { return $oneDriveDesktop }

  throw "Could not locate Desktop. Set -OutDir explicitly."
}

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

function Copy-IfExists([string]$Source, [string]$Dest) {
  if (Test-Path $Source) {
    $parent = Split-Path -Parent $Dest
    if ($parent) { Ensure-Dir $parent }
    Copy-Item -Force $Source $Dest
  }
}

function Copy-Tree([string]$SourceDir, [string]$DestDir, [string[]]$ExcludeDirs = @(), [string[]]$ExcludeFiles = @()) {
  if (-not (Test-Path $SourceDir)) { return }
  Ensure-Dir $DestDir

  $items = Get-ChildItem -LiteralPath $SourceDir -Force
  foreach ($item in $items) {
    if ($item.PSIsContainer) {
      if ($ExcludeDirs -contains $item.Name) { continue }
      Copy-Tree -SourceDir $item.FullName -DestDir (Join-Path $DestDir $item.Name) -ExcludeDirs $ExcludeDirs -ExcludeFiles $ExcludeFiles
    } else {
      if ($ExcludeFiles -contains $item.Name) { continue }
      Copy-Item -Force $item.FullName (Join-Path $DestDir $item.Name)
    }
  }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not $OutDir.Trim()) {
  $OutDir = Join-Path (Resolve-DesktopPath) "dwai-assistant_security_package_$stamp"
}

Ensure-Dir $OutDir

# Top-level files safe for review.
Copy-IfExists (Join-Path $repoRoot ".gitattributes") (Join-Path $OutDir ".gitattributes")
Copy-IfExists (Join-Path $repoRoot ".gitignore") (Join-Path $OutDir ".gitignore")
Copy-IfExists (Join-Path $repoRoot "docker-compose.yml") (Join-Path $OutDir "docker-compose.yml")
Copy-IfExists (Join-Path $repoRoot "LICENSE") (Join-Path $OutDir "LICENSE")
Copy-IfExists (Join-Path $repoRoot "README.md") (Join-Path $OutDir "README.md")

Get-ChildItem -LiteralPath $repoRoot -File -Filter "requirements*.txt" | ForEach-Object {
  Copy-IfExists $_.FullName (Join-Path $OutDir $_.Name)
}

# App code.
Copy-Tree -SourceDir (Join-Path $repoRoot "scripts") -DestDir (Join-Path $OutDir "scripts") -ExcludeDirs @("__pycache__")
Copy-Tree -SourceDir (Join-Path $repoRoot "webui") -DestDir (Join-Path $OutDir "webui") -ExcludeDirs @("__pycache__")

# Config + minimal data (no embeddings/index by default).
Copy-IfExists (Join-Path $repoRoot "scripts\\rag_config.json") (Join-Path $OutDir "scripts\\rag_config.json")
Copy-IfExists (Join-Path $repoRoot "scripts\\rag_config.example.json") (Join-Path $OutDir "scripts\\rag_config.example.json")

Ensure-Dir (Join-Path $OutDir "data")
Copy-IfExists (Join-Path $repoRoot "data\\rag\\config.json") (Join-Path $OutDir "data\\rag\\config.json")

if ($IncludeIndex) {
  # These can be very large.
  Copy-IfExists (Join-Path $repoRoot "data\\rag\\index.bin") (Join-Path $OutDir "data\\rag\\index.bin")
  Copy-IfExists (Join-Path $repoRoot "data\\rag\\meta.sqlite") (Join-Path $OutDir "data\\rag\\meta.sqlite")
  Copy-IfExists (Join-Path $repoRoot "data\\rag\\meta.jsonl") (Join-Path $OutDir "data\\rag\\meta.jsonl")
}

if ($IncludeFeedback) {
  Copy-IfExists (Join-Path $repoRoot "data\\feedback\\feedback.jsonl") (Join-Path $OutDir "data\\feedback\\feedback.jsonl")
}

if ($IncludeDocs) {
  # Warning: likely proprietary + huge; include only if your security team requests it.
  Copy-Tree -SourceDir (Join-Path $repoRoot "docs") -DestDir (Join-Path $OutDir "docs")
}

# Add an inventory for reviewers.
$inventoryPath = Join-Path $OutDir "INVENTORY.txt"
Get-ChildItem -LiteralPath $OutDir -Recurse -File | ForEach-Object {
  $rel = $_.FullName.Substring($OutDir.Length).TrimStart("\\")
  "{0}`t{1}" -f $rel, $_.Length
} | Sort-Object | Set-Content -Encoding UTF8 $inventoryPath

# Add a short review note.
$notesPath = Join-Path $OutDir "SECURITY_REVIEW.md"
$notes = @'
# dwai-assistant security review bundle

This bundle includes the application code and configs needed to review the project structure.

## What it does
- Builds a local RAG index from PDFs under `docs/` (not included by default).
- Serves a Streamlit Web UI on port `8501`.
- Uses a local Ollama server on port `11434` for embeddings and chat.

## Network behavior
- During normal use, the app calls `OLLAMA_HOST` / `ollama_base_url` (default `http://localhost:11434`).
- No other outbound network calls are required by the app at runtime.

## Exclusions by default
- `.venv/` (local Python environment)
- `docs/` (potentially proprietary PDFs)
- `data/rag/index.bin`, `data/rag/meta.*` (large generated index files)
- `data/feedback/feedback.jsonl` (may contain internal notes)

Regenerate with flags if needed:
`scripts\\package_for_security_review.ps1 -IncludeDocs -IncludeIndex -IncludeFeedback`
'@
$notes | Set-Content -Encoding UTF8 $notesPath

if ($Zip) {
  $zipPath = "$OutDir.zip"
  if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
  Compress-Archive -Path $OutDir -DestinationPath $zipPath -Force
  Write-Output "Created: $zipPath"
} else {
  Write-Output "Created folder: $OutDir"
}
