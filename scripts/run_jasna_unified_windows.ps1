[CmdletBinding()]
param(
    [string] $Python = "",
    [string] $RuntimeRoot = "",
    [switch] $PreflightOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $JasnaArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $repoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Jasna Python environment not found: $Python"
}

$launcherArgs = @()
if ($RuntimeRoot) {
    $launcherArgs += @("--runtime-root", $RuntimeRoot)
}
if ($PreflightOnly) {
    $launcherArgs += "--preflight-only"
}
if ($JasnaArgs.Count -gt 0) {
    $launcherArgs += "--"
    $launcherArgs += $JasnaArgs
}

& $Python (Join-Path $PSScriptRoot "run_jasna_unified.py") @launcherArgs
exit $LASTEXITCODE
