# PowerShell script to sync service documents from source to repo
# Usage: .\scripts\sync_service_docs.ps1

param(
    [string]$Source = "C:\Users\austin\Downloads\Service Documents\Service Documents",
    [string]$Dest = "$PSScriptRoot/../docs/service-documents",
    [switch]$Force = $false
)

# Resolve paths
$Source = Resolve-Path $Source -ErrorAction Stop
$Dest = if ([IO.Path]::IsPathRooted($Dest)) { $Dest } else { Join-Path (Split-Path $PSScriptRoot) $Dest }

Write-Host "DWAI Assistant - Service Documents Sync Script"
Write-Host "=" * 60
Write-Host "Source: $Source"
Write-Host "Destination: $Dest"
Write-Host ""

# Verify source exists
if (-not (Test-Path $Source -PathType Container)) {
    Write-Host "ERROR: Source path does not exist: $Source" -ForegroundColor Red
    exit 1
}

# Create destination if it doesn't exist
if (-not (Test-Path $Dest -PathType Container)) {
    Write-Host "Creating destination directory: $Dest"
    New-Item -ItemType Directory -Path $Dest -Force | Out-Null
}

# Robocopy parameters
# /E = Copy subdirectories (including empty ones)
# /XO = Exclude older files (only copy if newer)
# /R:1 = Retry once on failed files
# /W:1 = Wait 1 second between retries
# /NJH /NJS = No job header/summary (cleaner output)
# /NS /NC /NDCOPY /NP = No file sizes/classes/timestamp/progress percentage
# /V = Verbose (optional, comment out for less verbose)

$robocopyArgs = @(
    "`"$Source`"",
    "`"$Dest`"",
    "/E",
    "/XO",
    "/R:1",
    "/W:1",
    "/NJH",
    "/NJS"
)

if ($Force) {
    Write-Host "Running in FORCE mode (will overwrite all files)"
    # Remove /XO to force overwrite
    $robocopyArgs = $robocopyArgs -notmatch "/XO"
}

Write-Host "Starting sync with robocopy..."
Write-Host "-" * 60

# Execute robocopy
& robocopy $robocopyArgs

# Robocopy exit codes: 0-7 are success, 8+ are warnings/errors
$exitCode = $LASTEXITCODE
if ($exitCode -gt 7) {
    Write-Host ""
    Write-Host "WARNING: robocopy exited with code $exitCode (some files may not have copied)" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Sync completed successfully!" -ForegroundColor Green
}

Write-Host "-" * 60
Write-Host ""

# Summary
Write-Host "Summary:"
Write-Host "--------"
Write-Host "Files copied to: $Dest"

$fileCount = (Get-ChildItem -Path $Dest -Recurse -File).Count
Write-Host "Total files in destination: $fileCount"

Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Review the synced files: $Dest"
Write-Host "2. Commit changes to git:"
Write-Host "   cd $(Split-Path -Parent $Dest)"
Write-Host "   git add $(Split-Path -Leaf $Dest)"
Write-Host "   git commit -m 'Sync service documents'"
Write-Host "   git push origin main"
Write-Host "3. Reindex in Open WebUI: Settings → Datasets → DWAI Service Documents → Reindex"
