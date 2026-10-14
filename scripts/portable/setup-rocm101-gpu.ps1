<#
=============================================================================
 Jasna - ROCm 10.1 GPU setup / repair  (portable package, any Radeon)
=============================================================================

 WHY THIS SCRIPT EXISTS

 Windows ROCm wheels split the aotriton payload across TWO device packages:

   amd-torch-device-<exact arch>    ->  torch/.kpack/torch_<arch>.kpack
                                        (the actual torch kernels)
   amd-torch-device-<arch family>   ->  torch/lib/aotriton.images/<family>/...
                                        (the SDPA flash-attention images)

   * only the exact package  -> every kernel launch fails:
                                hipErrorInvalidKernelFile
   * only the family package -> also broken, its kpack name is not recognised
   * BOTH installed          -> works, and FLASH/EFFICIENT SDPA becomes usable

 That last point matters for speed: without the aotriton images PyTorch cannot
 launch the flash kernel, jasna falls back to the MATH backend (see
 jasna/accelerator.py), and RF-DETR's DINOv2 backbone runs ~2x slower - on a
 20 s clip that was 42.9 s -> 32.8 s end to end. Upstream tracking:
 ROCm/TheRock#7315.

 WHAT IT DOES (idempotent, safe to re-run)

   1. detects which gfx arch this GPU is (hipInfo, then a name table fallback)
   2. installs  amd-torch-device-<arch>  (+ amd-torchvision-device-<arch>)
   3. finds the family package that carries the aotriton images, installs it
      for the images and/or extracts torch/lib/aotriton.images from it
   4. reports whether FLASH SDPA is now available

 USAGE

   :: dry run - shows what would happen, changes nothing
   setup-rocm101-gpu.ps1 -CheckOnly

   :: normal (auto-detects the GPU)
   setup-rocm101-gpu.ps1

   :: force an arch (use when auto-detection fails or is unavailable)
   setup-rocm101-gpu.ps1 -Gfx gfx1101

   :: install from the wheels/ folder next to this script (offline)
   setup-rocm101-gpu.ps1 -WheelDir .\wheels

 ARCH NAMES (Radeon, for reference - the script detects this for you)

   gfx1100  RX 7900 XTX / 7900 XT / 7900 GRE / W7900
   gfx1101  RX 7800 / 7800 XT / 7700 / 7700 XT / W7800
   gfx1102  RX 7650 GRE / 7600 / 7600 XT / W7600
   gfx1103  iGPU (Phoenix)
   gfx1200  RX 9060 XT
   gfx1201  RX 9070 XT / 9070 / 9070 GRE
   gfx1151  Strix Halo / Ryzen AI Max
   gfx1030  RX 6800/6900 (RDNA2 - no flash support upstream, safe to skip)
#>

param(
    [string]$Venv     = '',
    [string]$Gfx      = '',
    [string]$WheelDir = '',
    [string]$Index    = 'https://stable.repo.amd.com/rocm/whl-next/',
    [switch]$CheckOnly,
    [switch]$SkipImages
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

# --- locate the bundled venv ------------------------------------------------
if (-not $Venv) {
    $pkgRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $Venv = Join-Path $pkgRoot 'venv'
}
$Py = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $Py)) { throw "Bundled python not found: $Py" }
$SitePackages = Join-Path $Venv 'Lib\site-packages'

function Write-Step($msg) { Write-Host ('  ' + $msg) }
function Write-Head($msg) { Write-Host ''; Write-Host ('== ' + $msg) }

Write-Host '====================================================='
Write-Host ' Jasna - ROCm 10.1 GPU setup / repair'
Write-Host '====================================================='
Write-Step ('venv  : ' + $Venv)
Write-Step ('index : ' + $Index)

# pip must not see an inherited PYTHONPATH (it breaks uninstalls)
$savedPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = ''

try {
    # --- 0) what is installed right now -------------------------------------
    Write-Head 'installed stack'
    $versions = & $Py -c @"
import json
try:
    import torch, torchvision
    print(json.dumps({
        'torch': torch.__version__,
        'hip': torch.version.hip or '',
        'torchvision': torchvision.__version__,
    }))
except Exception as exc:
    print(json.dumps({'error': str(exc)[:200]}))
"@ 2>$null | Select-Object -Last 1
    try { $info = $versions | ConvertFrom-Json } catch { $info = $null }
    if (-not $info -or $info.error) {
        throw ('Could not read torch version from the venv: ' + $versions)
    }
    $torchVer = $info.torch
    $tvVer    = $info.torchvision
    Write-Step ('torch       ' + $torchVer + '   (hip ' + $info.hip + ')')
    Write-Step ('torchvision ' + $tvVer)

    $torchBase = ($torchVer -split '\+')[0]          # 2.14.0
    $localTag  = ($torchVer -split '\+')[1]          # rocm10.1.0
    if (-not $localTag -or $localTag -notmatch '^rocm') {
        throw ('This venv is not a ROCm build (torch ' + $torchVer + '). Nothing to do.')
    }
    Write-Step ('local version tag: ' + $localTag)

    # --- 1) which GPU arch ---------------------------------------------------
    Write-Head 'GPU arch'
    if (-not $Gfx) {
        $hipInfo = Join-Path $Venv 'Scripts\hipInfo.exe'
        if (Test-Path $hipInfo) {
            $out = & $hipInfo 2>$null | Out-String
            if ($out -match 'gcnArchName[:\s]+(gfx[0-9a-f]+)') { $Gfx = $Matches[1] }
        }
        if (-not $Gfx) {
            $names = (Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                      Where-Object { $_.Name -match 'Radeon|AMD' } | Select-Object -ExpandProperty Name) -join ' '
            $table = @(
                @('RX 7900 XTX|RX 7900 XT|RX 7900 GRE|W7900|W7900D', 'gfx1100'),
                @('RX 7800|RX 7700|W7800',                            'gfx1101'),
                @('RX 7650|RX 7600',                                  'gfx1102'),
                @('RX 9070|RX 9070 GRE',                              'gfx1201'),
                @('RX 9060',                                          'gfx1200'),
                @('Strix Halo|Radeon 80[0-9]S|Radeon 8050S',          'gfx1151'),
                @('RX 6900|RX 6800',                                  'gfx1030')
            )
            foreach ($row in $table) { if ($names -match $row[0]) { $Gfx = $row[1]; break } }
            if ($Gfx) { Write-Step ('detected from driver name: ' + $names) }
            elseif ($names) { Write-Step ('AMD display adapters found, none matched the arch table: ' + $names) }
        }
    }
    if (-not $Gfx) {
        throw ('Could not detect the GPU arch. Re-run with -Gfx <arch>, e.g. -Gfx gfx1100 ' +
               '(see the arch table in this script).')
    }

    # --- iGPU-only safeguard -------------------------------------------------
    # If we landed on an iGPU arch (Phoenix gfx1103 / gfx1150) while the machine
    # also has a discrete Radeon whose driver never reported itself, the ROCm
    # runtime would target the wrong adapter. gfx1151 (Strix Halo) is a real
    # target arch, so it is excluded from this advice.
    if ($Gfx -match '^gfx1103$|^gfx1150$') {
        $hasDiscrete = $false
        $ctrls = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        foreach ($c in $ctrls) {
            if ($c.Name -match 'Radeon|AMD' -and $c.Name -notmatch 'Radeon \(TM\)|Graphics|Radeon 6[0-9]{2}M|Radeon 7[0-9]{2}M') { $hasDiscrete = $true }
        }
        # driver-name fallback found nothing discrete AND hipInfo still says iGPU
        if (-not $hasDiscrete) {
            Write-Head 'iGPU detected, no discrete AMD GPU'
            Write-Step ('hipInfo reports arch ' + $Gfx + ' (an integrated GPU).')
            Write-Step 'If this machine HAS a discrete Radeon (RX 6000/7000/9000 series) but it'
            Write-Step 'is not being seen, the iGPU is usually taking over. Try, in order:'
            Write-Step '  1. update the discrete GPU driver (amd.com/adrenalin), reboot, re-run;'
            Write-Step '  2. check BIOS: make sure the PCIe/Discrete GPU is enabled and the'
            Write-Step '     primary display is set to the discrete card;'
            Write-Step '  3. last resort: Device Manager -> Display adapters -> right-click the'
            Write-Step '     integrated GPU (e.g. AMD Radeon(tm) Graphics) -> Disable device,'
            Write-Step '     reboot, then re-run this script.'
            Write-Step ''
            Write-Step 'If this is genuinely an iGPU-only machine (Phoenix APU), ignore the advice'
            Write-Step 'above - gfx1103 is a supported arch and setup will continue below.'
        }
    }

    Write-Step ('gfx arch: ' + $Gfx)

    # --- 2) the exact-arch device packages ----------------------------------
    # The exact package carries torch/.kpack/torch_<arch>.kpack. Without it every
    # kernel launch raises hipErrorInvalidKernelFile.
    $exactPkgs = @(
        ('amd-torch-device-' + $Gfx + '==' + $torchVer)
    )
    # torchvision device package only exists for the arches torchvision supports
    $tvExact = 'amd-torchvision-device-' + $Gfx + '==' + $tvVer

    Write-Head 'exact-arch device packages'
    Write-Step ('amd-torch-device-' + $Gfx + '==' + $torchVer)
    Write-Step ($tvExact)

    # --- 3) the aotriton image package (arch family) ------------------------
    Write-Head 'aotriton image package (family)'
    $familyCandidates = @()
    if ($Gfx -match '^gfx(\d)(\d)(\d)(\d)$') {
        # gfx1100 -> gfx110x / gfx11x ; gfx1201 -> gfx12-0 / gfx120x / gfx12x
        $familyCandidates += ('gfx' + $Matches[1] + $Matches[2] + $Matches[3] + 'x')
        $familyCandidates += ('gfx' + $Matches[1] + $Matches[2] + 'x')
        if (($Matches[1] + $Matches[2]) -eq '12') { $familyCandidates += 'gfx12-0' }
        $familyCandidates += ('gfx' + $Matches[1] + $Matches[2] + $Matches[3])
    }
    $familyCandidates = $familyCandidates | Where-Object { $_ -and $_ -ne $Gfx } | Select-Object -Unique
    Write-Step ('candidates: ' + ($familyCandidates -join ', '))

    # which of them really exist on the index?
    $available = @()
    try {
        $listing = & curl.exe -sS --max-time 30 $Index 2>$null | Out-String
        foreach ($c in $familyCandidates) {
            if ($listing -match ('amd-torch-device-' + [regex]::Escape($c) + '/')) { $available += $c }
        }
    } catch {
        Write-Step ('index listing unavailable, will try candidates blindly')
    }
    if ($available.Count -gt 0) { Write-Step ('available  : ' + ($available -join ', ')) }
    $tryList = if ($available.Count -gt 0) { $available } else { $familyCandidates }

    if ($CheckOnly) {
        Write-Head 'DRY RUN - nothing will be installed'
        $dry = @($exactPkgs) + @($tvExact)
        & $Py -m pip install --dry-run --disable-pip-version-check --no-input --index-url $Index @dry
        Write-Step ''
        Write-Step 'The image package cannot be dry-run reliably; the real run downloads'
        Write-Step 'each candidate until one contains torch/lib/aotriton.images.'
        return
    }

    # --- 4) install ---------------------------------------------------------
    $bak = Join-Path $Venv ('pip_freeze_before_' + $Gfx + '.txt')
    & $Py -m pip freeze 2>$null | Out-File -FilePath $bak -Encoding ascii
    Write-Head 'installing exact-arch device packages'
    Write-Step ('backup: ' + $bak)
    $installArgs = @($exactPkgs) + @($tvExact)
    & $Py -m pip install --disable-pip-version-check --no-input --index-url $Index @installArgs
    if ($LASTEXITCODE -ne 0) { Write-Step 'WARN: pip returned a non-zero exit code' }

    $imagesRoot = Join-Path $SitePackages 'torch\lib\aotriton.images'
    $gotImages = Test-Path (Join-Path $imagesRoot '*\flash\attn_fwd.zip')

    if (-not $SkipImages -and -not $gotImages) {
        Write-Head 'aotriton images'
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $tmp = Join-Path $env:TEMP ('jasna_aotriton_' + $Gfx)
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null

        foreach ($fam in $tryList) {
            $pkg = 'amd-torch-device-' + $fam + '==' + $torchVer
            Write-Step ('trying ' + $pkg)
            $dl = Join-Path $tmp $fam
            New-Item -ItemType Directory -Force -Path $dl | Out-Null

            $localWheel = $null
            if ($WheelDir) {
                $famFile = $fam.Replace('-', '_')
                $localWheel = Get-ChildItem $WheelDir -Filter ('amd_torch_device_' + $famFile + '-*.whl') -ErrorAction SilentlyContinue |
                              Select-Object -First 1
            }
            if ($localWheel) {
                Write-Step ('  using local wheel ' + $localWheel.Name)
                $wheelPath = $localWheel.FullName
            } else {
                & $Py -m pip download --no-deps --disable-pip-version-check --no-input `
                      --index-url $Index -d $dl $pkg 2>$null | Out-Null
                $wheelPath = (Get-ChildItem $dl -Filter '*.whl' -ErrorAction SilentlyContinue |
                              Select-Object -First 1).FullName
            }
            if (-not $wheelPath) { Write-Step '  not available, next candidate'; continue }

            $zip  = [System.IO.Compression.ZipFile]::OpenRead($wheelPath)
            $imgs = @($zip.Entries | Where-Object { $_.FullName -like 'torch/lib/aotriton.images/*' -and $_.Length -gt 0 })
            if ($imgs.Count -eq 0) {
                Write-Step ('  no aotriton images inside (' + [math]::Round((Get-Item $wheelPath).Length/1MB,1) + ' MB), next candidate')
                $zip.Dispose()
                continue
            }
            $mb = [math]::Round((($imgs | Measure-Object Length -Sum).Sum) / 1MB, 1)
            Write-Step ('  extracting ' + $imgs.Count + ' files / ' + $mb + ' MB')
            foreach ($e in $imgs) {
                $target = Join-Path $SitePackages ($e.FullName -replace '/', '\')
                New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $target, $true)
            }
            $zip.Dispose()
            $gotImages = Test-Path (Join-Path $imagesRoot '*\flash\attn_fwd.zip')
            if ($gotImages) { Write-Step ('  OK - images installed from amd-torch-device-' + $fam); break }
        }
        if (-not $gotImages) {
            Write-Step 'WARN: no aotriton images found for this arch.'
            Write-Step '      jasna will keep working, but SDPA stays on the slow MATH path.'
            Write-Step '      Check the index listing manually and pass -WheelDir if needed.'
        }
    } elseif ($gotImages) {
        Write-Head 'aotriton images'
        Write-Step 'already present - nothing to do'
    }

    # --- 5) status (static checks only, no GPU work) ------------------------
    Write-Head 'status'
    $kpack = Join-Path $SitePackages ('torch\.kpack\torch_' + $Gfx + '.kpack')
    Write-Step ('kpack  torch_' + $Gfx + '.kpack : ' + $(if (Test-Path $kpack) { 'present' } else { 'MISSING' }))
    $imgs = Get-ChildItem $imagesRoot -Recurse -File -ErrorAction SilentlyContinue
    if ($imgs) {
        Write-Step ('aotriton images          : ' + $imgs.Count + ' files / ' +
                    [math]::Round((($imgs | Measure-Object Length -Sum).Sum) / 1MB, 1) + ' MB')
    } else {
        Write-Step 'aotriton images          : MISSING'
    }
    $check = & $Py -c @"
try:
    from jasna.accelerator import _aotriton_images_present
    print('flash usable:', _aotriton_images_present())
except Exception as exc:
    print('flash usable: unknown (' + str(exc)[:80] + ')')
"@ 2>$null | Select-Object -Last 1
    Write-Step $check
    & $Py -m pip check 2>$null
}
finally {
    $env:PYTHONPATH = $savedPythonPath
}

Write-Host ''
Write-Host 'DONE.'
Write-Host '  If "flash usable: True", the fast attention path is active.'
Write-Host '  To verify visually, run any clip and compare elapsed time before/after.'
Write-Host '  To force the old (slow but safe) behaviour: set JASNA_FORCE_MATH_SDP=1'
