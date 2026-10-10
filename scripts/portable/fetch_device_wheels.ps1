<#
=============================================================================
 Fetch the per-architecture ROCm device wheels a portable package must bundle.
=============================================================================

A portable tree is pinned to one GPU architecture: venv\ carries
torch\.kpack\torch_<arch>.kpack and torch\lib\aotriton.images\amd-<family>, and
wheels\ carries the matching amd-torch-device-<arch>, amd-torch-device-<family>,
amd-torchvision-device-<arch> and rocm-sdk-device-<arch> wheels.

A package assembled on a gfx1100 machine therefore cannot be installed on an
RX 9070 (gfx1201) without network access: the exact-arch kernel package, its
attention images and its rocBLAS / MIOpen / hipFft kernels are all missing, and
rocm-sdk-device-<arch> (a hard dependency of amd-torch-device-<arch>) is not in
wheels\. The one-click setup then has to reach AMD's index, which fails outright
on an offline machine - and a missing kernel package only shows up later as
"hipErrorInvalidKernelFile", while a missing attention image silently drops SDPA
onto the MATH path (roughly 2x slower).

Run this once per architecture you intend to support, with the venv of the
package that will be shipped, and wheels\ ends up complete:

    powershell -ExecutionPolicy Bypass -File fetch_device_wheels.ps1 -Gfx gfx1201
    powershell -ExecutionPolicy Bypass -File fetch_device_wheels.ps1 -Gfx gfx1100

AMD's index is a rolling one: older builds disappear from it. If a version the
venv needs is gone (the report says so), keep the copy you already have in
wheels\ instead of downloading something that does not match the installed torch.

.PARAMETER Gfx
  Target architecture: gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151,
  gfx1152, gfx1200, gfx1201. Defaults to the arch of the GPU in this machine
  (via hipInfo, falling back to torch).

.PARAMETER Package
  Package root (the folder that holds venv\, app\ and wheels\). Defaults to this
  script's parent folder's parent.

.PARAMETER Index
  AMD package index. Default https://stable.repo.amd.com/rocm/whl-next/

.PARAMETER OutDir
  Where the wheels are written. Default <Package>\wheels

.PARAMETER DryRun
  Resolve and report only; download nothing.

.PARAMETER Force
  Re-download wheels that are already present.

.PARAMETER Proxy
  Optional HTTP proxy for the downloads, e.g. http://127.0.0.1:7897
#>
param(
    [string]$Gfx = "",
    [string]$Package = "",
    [string]$Index = "https://stable.repo.amd.com/rocm/whl-next/",
    [string]$OutDir = "",
    [switch]$DryRun,
    [switch]$Force,
    [string]$Proxy = ""
)

$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

if (-not $Package) { $Package = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$venv = Join-Path $Package 'venv'
$py = Join-Path $venv 'Scripts\python.exe'
if (-not $OutDir) { $OutDir = Join-Path $Package 'wheels' }
if (-not (Test-Path $py)) { throw ('No venv python at ' + $py) }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

function Write-Step($text) { Write-Host ('  ' + $text) }
function Write-Head($text) { Write-Host ''; Write-Host ('== ' + $text + ' ' + ('-' * [Math]::Max(2, 54 - $text.Length))) }

function Get-IndexWheels([string]$name) {
    $url = $Index + $name + '/'
    try {
        $html = (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 60).Content
    } catch {
        Write-Step ('cannot list ' + $url + ' : ' + $_.Exception.Message)
        return @()
    }
    $out = @()
    foreach ($match in [regex]::Matches($html, 'href="([^"#]+\.whl)(?:#sha256=([0-9a-f]+))?"')) {
        # keep the href exactly as published: a local version separator '+' is encoded
        # as %2B there, and the server 404s on the decoded form
        $out += [pscustomobject]@{
            Href = $match.Groups[1].Value
            File = [uri]::UnescapeDataString($match.Groups[1].Value)
            Sha  = $match.Groups[2].Value
        }
    }
    return $out
}

function Get-PinnedVersion([string]$module, [string]$fallback) {
    $value = & $py -c ("import " + $module + " as m; print(m.__version__)") 2>$null | Select-Object -Last 1
    if ($value) { return $value.Trim() }
    return $fallback
}

# --- which architecture ------------------------------------------------------
Write-Head '1. Target architecture'
if (-not $Gfx) {
    $hipInfo = Join-Path $venv 'Scripts\hipInfo.exe'
    if (Test-Path $hipInfo) {
        $out = & $hipInfo 2>$null | Out-String
        if ($out -match 'gcnArchName[:\s]+(gfx[0-9a-f]+)') { $Gfx = $Matches[1] }
    }
}
if (-not $Gfx) {
    $Gfx = & $py -c "import torch;n=torch.cuda.device_count();print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] if n else '')" 2>$null | Select-Object -Last 1
    if ($Gfx) { $Gfx = $Gfx.Trim() }
}
if (-not $Gfx) { throw 'Cannot determine the architecture; pass -Gfx gfx1100 / gfx1201 / ...' }
if ($Gfx -notmatch '^gfx\d{4}$') { throw ('Unsupported -Gfx value: ' + $Gfx) }
# the attention images ship per family: gfx1100 -> gfx110x, gfx1201 -> gfx120x
$family = $Gfx.Substring(0, 6) + 'x'
Write-Step ('architecture : ' + $Gfx)
Write-Step ('attention family : ' + $family)

$torchVer = Get-PinnedVersion 'torch' '2.14.0+rocm10.1.0'
$tvVer = Get-PinnedVersion 'torchvision' '0.29.0a0+rocm10.1.0'
$cp = & $py -c "import sys;print('cp%d%d' % sys.version_info[:2])" 2>$null | Select-Object -Last 1
if ($cp) { $cp = $cp.Trim() } else { $cp = 'cp312' }
Write-Step ('installed torch : ' + $torchVer)
Write-Step ('installed torchvision : ' + $tvVer)
Write-Step ('interpreter tag : ' + $cp + '-' + $cp + '-win_amd64')

# rocm-sdk-device-<arch> is versioned like the rest of the ROCm SDK wheels
$sdkVer = '10.1.0'
$sdkLine = & $py -m pip show rocm-sdk-libraries 2>$null | Select-String '^Version:'
if ($sdkLine) { $sdkVer = ($sdkLine -split ':')[1].Trim() }
$deviceSdk = & $py -m pip show ('rocm-sdk-device-' + $Gfx) 2>$null | Select-String '^Version:'
$deviceSdkVer = if ($deviceSdk) { ($deviceSdk -split ':')[1].Trim() } else { $sdkVer }
Write-Step ('rocm-sdk device version : ' + $deviceSdkVer)

# --- what the package needs --------------------------------------------------
Write-Head '2. Wheels this architecture needs'
$wanted = @(
    [pscustomobject]@{ Name = 'rocm-sdk-device-' + $Gfx; Version = $deviceSdkVer },
    [pscustomobject]@{ Name = 'amd-torch-device-' + $Gfx; Version = $torchVer },
    [pscustomobject]@{ Name = 'amd-torch-device-' + $family; Version = $torchVer },
    [pscustomobject]@{ Name = 'amd-torchvision-device-' + $Gfx; Version = $tvVer }
)

$plan = @()
foreach ($want in $wanted) {
    # local wheel filenames keep the '+' of the local version, the index links encode it
    # as %2B, so compare against the unescaped name and the plain version everywhere
    $version = $want.Version
    $stem = ($want.Name -replace '-', '_')
    $present = $false
    foreach ($file in @(Get-ChildItem $OutDir -Filter ($stem + '-*.whl') -ErrorAction SilentlyContinue)) {
        if ($file.Name -like ('*' + $version + '*')) { $present = $true }
    }
    $candidates = Get-IndexWheels $want.Name
    $pick = $null
    foreach ($candidate in $candidates) {
        $name = Split-Path -Leaf $candidate.File
        if ($name -like '*linux*') { continue }
        # rocm-sdk-* wheels are py3-none-win_amd64, the torch ones carry the cp tag
        if ($name -notlike ('*' + $cp + '*') -and $name -notlike '*py3-none*') { continue }
        if ($name -notlike ('*' + $version + '*')) { continue }
        $pick = $candidate
        break
    }
    $matches = $false
    if ($pick) { $matches = $true } else {
        # nothing with the wanted version: report what the index still has instead
        foreach ($candidate in $candidates) {
            $name = Split-Path -Leaf $candidate.File
            if ($name -like '*linux*') { continue }
            if ($name -like ('*' + $cp + '*') -or $name -like '*py3-none*') { $pick = $candidate; break }
        }
    }
    $plan += [pscustomobject]@{
        Spec    = $want.Name + '==' + $want.Version
        Present = $present
        Pick    = $pick
        Matches = $matches
    }
}

$missing = 0
foreach ($item in $plan) {
    if ($item.Present) {
        Write-Step ('present : ' + $item.Spec)
    } elseif ($item.Pick -and $item.Matches) {
        Write-Step ('missing : ' + $item.Spec + '  ->  ' + (Split-Path -Leaf $item.Pick.File))
        $missing++
    } elseif ($item.Pick) {
        Write-Step ('missing : ' + $item.Spec + '  ->  index only has ' + (Split-Path -Leaf $item.Pick.File))
        $missing++
    } else {
        Write-Step ('missing : ' + $item.Spec + '  ->  not on the index at all')
        $missing++
    }
}
if ($missing -eq 0) {
    Write-Host ''
    Write-Host ('  wheels\ already covers ' + $Gfx + ' - nothing to fetch.') -ForegroundColor Green
} elseif ($DryRun) {
    Write-Host ''
    Write-Host ('  -DryRun: ' + $missing + ' wheel(s) would be fetched.') -ForegroundColor Yellow
} else {
    Write-Host ''
    Write-Host ('  fetching ' + $missing + ' wheel(s) into ' + $OutDir) -ForegroundColor Cyan
}

# --- download ----------------------------------------------------------------
if (-not $DryRun -and $missing -gt 0) {
    Write-Head '3. Download'
    foreach ($item in $plan) {
        if ($item.Present) { continue }
        if (-not $item.Pick) {
            Write-Step ('skip ' + $item.Spec + ' (nothing usable on the index)')
            continue
        }
        $url = $Index + ($item.Spec -split '==' | Select-Object -First 1) + '/' + $item.Pick.Href
        $target = Join-Path $OutDir (Split-Path -Leaf $item.Pick.File)
        if ((Test-Path $target) -and -not $Force) {
            Write-Step ('keep ' + (Split-Path -Leaf $target))
        } else {
            $curl = (Get-Command curl.exe -ErrorAction SilentlyContinue)
            if ($curl) {
                $curlArgs = @('-L', '--fail', '--retry', '3', '-o', $target, $url)
                if ($Proxy) { $curlArgs = @('-x', $Proxy) + $curlArgs }
                Write-Step ('curl ' + (Split-Path -Leaf $item.Pick.File))
                & $curl.Source @curlArgs
            } else {
                Write-Step ('Invoke-WebRequest ' + (Split-Path -Leaf $item.Pick.File))
                $iwrArgs = @{ Uri = $url; OutFile = $target; UseBasicParsing = $true; TimeoutSec = 1800 }
                if ($Proxy) { $iwrArgs['Proxy'] = $Proxy }
                Invoke-WebRequest @iwrArgs
            }
        }
        if ((Test-Path $target) -and $item.Pick.Sha) {
            $hash = (Get-FileHash $target -Algorithm SHA256).Hash.ToLower()
            if ($hash -ne $item.Pick.Sha) {
                Write-Host ('  [!!] checksum mismatch for ' + (Split-Path -Leaf $target)) -ForegroundColor Red
                Write-Host ('       expected ' + $item.Pick.Sha) -ForegroundColor Red
                Write-Host ('       got      ' + $hash) -ForegroundColor Red
            } else {
                Write-Step ('sha256 ok : ' + $hash.Substring(0, 16))
            }
        }
    }
}

# --- report ------------------------------------------------------------------
Write-Head '4. Result'
$summary = @()
foreach ($file in @(Get-ChildItem $OutDir -Filter '*.whl' -ErrorAction SilentlyContinue | Sort-Object Name)) {
    $mb = [Math]::Round($file.Length / 1MB, 1)
    Write-Step (('{0,8} MB  {1}' -f $mb, $file.Name))
    $summary += (('{0}  {1}' -f (Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLower(), $file.Name))
}
if ($DryRun) { return }
$note = Join-Path $OutDir 'wheels-说明.txt'
$lines = @(
    ('target architecture : ' + $Gfx),
    ('attention family    : ' + $family),
    ('torch / torchvision : ' + $torchVer + ' / ' + $tvVer),
    ('index               : ' + $Index),
    '',
    'A portable package must bundle these for this architecture:',
    ('  amd-torch-device-' + $Gfx + '            -> torch\.kpack\torch_' + $Gfx + '.kpack (kernels)'),
    ('  amd-torch-device-' + $family + '            -> torch\lib\aotriton.images\amd-' + $family + ' (SDPA images)'),
    ('  rocm-sdk-device-' + $Gfx + '            -> rocBLAS / MIOpen / hipFft kernels'),
    ('  amd-torchvision-device-' + $Gfx + '       -> torchvision kernels'),
    '',
    'AMD''s index is a rolling one: builds disappear from it. If the version the',
    'venv needs is gone, keep the copy already in wheels\ instead of fetching a',
    'build that does not match the installed torch.',
    '',
    'sha256:'
) + $summary
Set-Content -Path $note -Value $lines -Encoding UTF8
Write-Step ('wrote ' + $note)
