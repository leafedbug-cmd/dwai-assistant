param(
  [string]$SourceRoot = "C:\Users\atruett\Box\Service Documents",
  [string]$DestRoot = "C:\Users\atruett\Documents\FastSearch\dwai-assistant\docs",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $SourceRoot)) {
  throw "SourceRoot not found: $SourceRoot"
}

New-Item -ItemType Directory -Force -Path $DestRoot | Out-Null

# Exclude obvious non-English language markers in either folder names or filenames.
# This is heuristic: it copies everything UNLESS it looks explicitly non-English.
$excludeTokens = @(
  "spanish", "espanol", "español", "castellano",
  "french", "francais", "français",
  "german", "deutsch",
  "italian", "italiano",
  "portuguese", "portugues", "português", "brazil", "brasil",
  "dutch", "nederlands",
  "swedish", "svenska",
  "norwegian", "norsk",
  "danish", "dansk",
  "finnish", "suomi",
  "polish", "polski",
  "czech", "cesky", "česky",
  "slovak", "slovensky",
  "hungarian", "magyar",
  "romanian", "română", "romana",
  "bulgarian", "български",
  "greek", "ελληνικά",
  "russian", "русский",
  "ukrainian", "українська",
  "turkish", "türkçe",
  "arabic", "العربية",
  "hebrew", "עברית",
  "chinese", "中文", "mandarin",
  "japanese", "日本語",
  "korean", "한국어",
  "thai", "ไทย",
  "vietnamese", "tiếng việt"
)

$excludeRegex = "(?i)(" + (($excludeTokens | ForEach-Object { [regex]::Escape($_) }) -join "|") + ")"

function ShouldExcludePath([string]$relativePath) {
  return [regex]::IsMatch($relativePath, $excludeRegex)
}

$sourceRootResolved = (Resolve-Path -LiteralPath $SourceRoot).Path.TrimEnd("\")
$destRootResolved = (Resolve-Path -LiteralPath $DestRoot).Path.TrimEnd("\")

$total = 0
$dirErrors = 0
$excluded = 0
$copied = 0
$skippedUpToDate = 0

function CopyIfNeeded([System.IO.FileInfo]$srcFile) {
  $script:total++
  $rel = $srcFile.FullName.Substring($sourceRootResolved.Length).TrimStart("\")

  if (ShouldExcludePath($rel)) {
    $script:excluded++
    return
  }

  $dst = Join-Path $destRootResolved $rel
  $dstDir = Split-Path -Path $dst -Parent
  if (-not (Test-Path -LiteralPath $dstDir)) {
    if (-not $DryRun) {
      New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    }
  }

  $shouldCopy = $true
  if (Test-Path -LiteralPath $dst) {
    try {
      $dstItem = Get-Item -LiteralPath $dst
      if (($dstItem.Length -eq $srcFile.Length) -and ($dstItem.LastWriteTimeUtc -ge $srcFile.LastWriteTimeUtc)) {
        $shouldCopy = $false
      }
    } catch {
      $shouldCopy = $true
    }
  }

  if (-not $shouldCopy) {
    $script:skippedUpToDate++
    return
  }

  if ($DryRun) {
    Write-Output ("DRYRUN copy: {0} -> {1}" -f $srcFile.FullName, $dst)
  } else {
    Copy-Item -LiteralPath $srcFile.FullName -Destination $dst -Force
  }
  $script:copied++
}

# Manual directory walk so a single bad Box folder doesn't abort the whole sync.
$stack = New-Object System.Collections.Generic.Stack[string]
$stack.Push($sourceRootResolved)

while ($stack.Count -gt 0) {
  $dir = $stack.Pop()

  $relDir = $dir.Substring($sourceRootResolved.Length).TrimStart("\")
  if ($relDir -and (ShouldExcludePath($relDir))) {
    $excluded++
    continue
  }

  try {
    $children = Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop
  } catch {
    $dirErrors++
    Write-Warning ("Skipping unreadable folder: {0} :: {1}" -f $dir, $_.Exception.Message)
    continue
  }

  foreach ($c in $children) {
    if ($c.PSIsContainer) {
      $stack.Push($c.FullName)
    } elseif ($c -is [System.IO.FileInfo]) {
      CopyIfNeeded $c
    }
  }
}

Write-Output ""
Write-Output ("Source: {0}" -f $sourceRootResolved)
Write-Output ("Dest:   {0}" -f $destRootResolved)
Write-Output ("Total files scanned:     {0}" -f $total)
Write-Output ("Unreadable folders:      {0}" -f $dirErrors)
Write-Output ("Excluded (non-English):  {0}" -f $excluded)
Write-Output ("Copied/updated:          {0}" -f $copied)
Write-Output ("Skipped (up-to-date):    {0}" -f $skippedUpToDate)
