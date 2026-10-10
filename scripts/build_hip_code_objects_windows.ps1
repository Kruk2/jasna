[CmdletBinding()]
param(
    [string] $Architecture = "gfx1100",
    [string] $Hipcc = "",
    [string] $Python = "python",
    [string] $OutputDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$mediaDirectory = Join-Path $repoRoot "jasna\media"
if (-not $OutputDirectory) {
    $OutputDirectory = $mediaDirectory
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path

if (-not $Hipcc) {
    if ($env:HIP_PATH) {
        $Hipcc = Join-Path $env:HIP_PATH "bin\hipcc.exe"
    } else {
        $Hipcc = (Get-Command hipcc.exe -ErrorAction Stop).Source
    }
}
$Hipcc = (Resolve-Path -LiteralPath $Hipcc -ErrorAction Stop).Path
$Python = (& (Get-Command $Python -ErrorAction Stop).Source -c "import sys; print(sys.executable)").Trim()
$sdkRoot = Split-Path -Parent (Split-Path -Parent $Hipcc)
if (-not (Test-Path -LiteralPath (Join-Path $sdkRoot "include\hip\hip_runtime.h") -PathType Leaf)) {
    throw "Cannot determine a complete HIP SDK root from $Hipcc"
}
$previousHipPath = $env:HIP_PATH
$previousRocmPath = $env:ROCM_PATH
try {
    # hipcc consults these variables even when invoked by an absolute path.
    # Pin both to the SDK that owns the selected compiler so an ambient system
    # installation cannot mix headers/device libraries from another release.
    $env:HIP_PATH = $sdkRoot
    $env:ROCM_PATH = $sdkRoot

$compilerVersion = (& $Hipcc --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $compilerVersion -notmatch 'HIP version:\s*([0-9]+)\.') {
    throw "Cannot determine HIP compiler major from $Hipcc"
}
$compilerMajor = [int]$Matches[1]
$runtimeJson = & $Python -c @"
import ctypes, hashlib, json, pathlib, torch
device = torch.device('cuda:0')
if not torch.cuda.is_available() or torch.version.hip is None:
    raise SystemExit('a ROCm-backed PyTorch runtime is required')
torch.cuda.init()
major = int(str(torch.version.hip).split('.', 1)[0])
dll_name = f'amdhip64_{major}.dll'
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
kernel32.GetModuleHandleW.restype = ctypes.c_void_p
kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
kernel32.GetModuleFileNameW.restype = ctypes.c_uint
handle = kernel32.GetModuleHandleW(dll_name)
buffer = ctypes.create_unicode_buffer(32768)
if not handle or not kernel32.GetModuleFileNameW(handle, buffer, len(buffer)):
    raise SystemExit(f'cannot resolve PyTorch HIP runtime {dll_name}')
runtime_path = pathlib.Path(buffer.value)
lib = ctypes.WinDLL(dll_name, handle=handle, use_last_error=True)
lib.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.hipRuntimeGetVersion.restype = ctypes.c_int
runtime_api = ctypes.c_int()
if lib.hipRuntimeGetVersion(ctypes.byref(runtime_api)) != 0:
    raise SystemExit('hipRuntimeGetVersion failed')
print(json.dumps({
    'hip': torch.version.hip,
    'architecture': str(getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '')).split(':', 1)[0],
    'runtime_dll': dll_name,
    'runtime_dll_sha256': hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
    'runtime_api_version': runtime_api.value,
}))
"@
if ($LASTEXITCODE -ne 0) {
    throw "Cannot inspect the PyTorch HIP runtime with $Python"
}
$runtime = $runtimeJson | ConvertFrom-Json
if ([string]$runtime.hip -notmatch '^([0-9]+)\.') {
    throw "Cannot determine PyTorch HIP major from $($runtime.hip)"
}
$runtimeMajor = [int]$Matches[1]
if ($compilerMajor -ne $runtimeMajor) {
    throw "HIP compiler major $compilerMajor does not match PyTorch HIP major $runtimeMajor"
}
if ([string]$runtime.architecture -ne $Architecture) {
    throw "Requested architecture $Architecture does not match PyTorch device $($runtime.architecture)"
}

$sources = @("yuv_to_rgb.cu", "rgb_to_yuv.cu")
$deviceLibraryDirectory = Join-Path $sdkRoot "lib\llvm\amdgcn\bitcode"
if (-not (Test-Path -LiteralPath (Join-Path $deviceLibraryDirectory "ocml.bc") -PathType Leaf)) {
    throw "ROCm device libraries are missing from the selected HIP SDK: $sdkRoot"
}
$artifactHashes = [ordered]@{}
$sourceHashes = [ordered]@{}
Push-Location $mediaDirectory
try {
    foreach ($sourceName in $sources) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($sourceName)
        $artifactName = "$stem.$Architecture.windows.co"
        $destination = Join-Path $OutputDirectory $artifactName
        $compileArgs = @(
            "--genco",
            "--no-gpu-bundle-output",
            "-fuse-cuid=none",
            "--offload-arch=$Architecture",
            "--rocm-device-lib-path=$deviceLibraryDirectory",
            "-O3",
            "-std=c++17",
            "-include",
            "hip/hip_runtime.h",
            $sourceName,
            "-o",
            $destination
        )
        & $Hipcc @compileArgs
        if ($LASTEXITCODE -ne 0) {
            throw "hipcc failed for $sourceName with exit $LASTEXITCODE"
        }
        $header = [IO.File]::ReadAllBytes($destination)
        if ($header[0] -ne 0x7f -or $header[1] -ne 0x45 -or $header[2] -ne 0x4c -or $header[3] -ne 0x46) {
            throw "$artifactName is not a raw ELF HIP code object"
        }
        # ELF64 little-endian, AMDGPU HSA OS ABI, code-object ABI v4,
        # EM_AMDGPU. gfx1100 is then protected by the compiler target plus the
        # manifest/artifact hash and independently checked by the runtime gate.
        if ($header[4] -ne 2 -or $header[5] -ne 1 -or $header[7] -ne 0x40 -or $header[8] -ne 4 -or $header[18] -ne 0xe0 -or $header[19] -ne 0) {
            throw "$artifactName does not declare the expected AMDGPU HSA code-object ABI"
        }
        $sourceHashes[$sourceName] = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceName).Hash.ToLower()
        $artifactHashes[$artifactName] = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLower()
    }
} finally {
    Pop-Location
}

$manifest = [ordered]@{
    schema = "jasna.hip-color-kernels.windows.v1"
    platform = "win32"
    architecture = $Architecture
    hip_major = $runtimeMajor
    compiler_hip_version = ($compilerVersion -split "`r?`n")[0]
    torch_hip_runtime = [string]$runtime.hip
    runtime_api_version = [int]$runtime.runtime_api_version
    runtime_dll = [string]$runtime.runtime_dll
    runtime_dll_sha256 = [string]$runtime.runtime_dll_sha256
    code_object_abi = 4
    parameter_abi = "jasna.hip-module-kernel-params.v1"
    compiler_arguments = @(
        "--genco",
        "--no-gpu-bundle-output",
        "-fuse-cuid=none",
        "--offload-arch=$Architecture",
        "--rocm-device-lib-path=<selected-sdk>/lib/llvm/amdgcn/bitcode",
        "-O3",
        "-std=c++17",
        "-include hip/hip_runtime.h"
    )
    sources = $sourceHashes
    artifacts = $artifactHashes
}
$manifestPath = Join-Path $OutputDirectory "hip_color_kernels.$Architecture.windows.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "Windows HIP colour kernels: $OutputDirectory"
Write-Host "Manifest: $manifestPath"
} finally {
    $env:HIP_PATH = $previousHipPath
    $env:ROCM_PATH = $previousRocmPath
}
