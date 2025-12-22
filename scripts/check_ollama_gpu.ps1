$ErrorActionPreference = "Stop"

function Get-TopOllamaProcess {
  $procs = Get-Process ollama -ErrorAction SilentlyContinue
  if (-not $procs) { return $null }
  return ($procs | Sort-Object CPU -Descending | Select-Object -First 1)
}

$oll = Get-TopOllamaProcess
if (-not $oll) {
  Write-Output "No ollama.exe process found."
  exit 1
}

$ollId = $oll.Id
Write-Output ("Top ollama PID: {0}" -f $ollId)
Write-Output ("Top ollama CPU seconds: {0}" -f $oll.CPU)
Write-Output ("Top ollama WorkingSet (GB): {0}" -f ([math]::Round($oll.WorkingSet64 / 1GB, 2)))

try {
  $samples = (Get-Counter "\\GPU Engine(*)\\Utilization Percentage" -SampleInterval 1 -MaxSamples 3).CounterSamples
} catch {
  Write-Output ("Failed to read GPU counters: {0}" -f $_.Exception.Message)
  exit 2
}

$hits =
  $samples |
  Where-Object { $_.InstanceName -match ("pid_{0}" -f $ollId) } |
  Group-Object InstanceName |
  ForEach-Object {
    $avg = ($_.Group | Measure-Object CookedValue -Average).Average
    [pscustomobject]@{
      Instance = $_.Name
      AvgUtilPercent = [math]::Round($avg, 2)
    }
  } |
  Sort-Object AvgUtilPercent -Descending

if (-not $hits) {
  Write-Output "No GPU Engine instances matched this ollama PID (may be CPU-only or counters unavailable)."
  exit 0
}

Write-Output ""
Write-Output "Top GPU Engine instances for this ollama PID:"
$hits | Select-Object -First 12 | Format-Table -AutoSize

