<#
=============================================================================
 Jasna 便携版 - 首次运行一键配置
=============================================================================

双击同目录的「首次运行-一键配置.bat」即可运行本脚本（它会自动传参）。
做四件事，全部可重复执行，已配置好的会被跳过：

  1. 配置包内 Python 虚拟环境（venv\pyvenv.cfg + 让 venv 找到 app\ 里的程序）
  2. 识别显卡与架构，检查 ROCm 内核包（缺失才安装，优先用包内 wheels\ 离线安装）
  3. 配置 MIGraphX 检测引擎的运行库：
       优先用 app\model_weights\migraphx-ep\ 内自带的副本（不依赖 C 盘、不依赖商店）
       没有则直接从 Windows ML 已安装的包里复制一份进来
       都没有则维持原样（运行时自动回退到 Windows ML / torch 路径）
  4. 自检：跑一次真实的检测引擎建会话，确认引擎真的能用

参数（可选）：
  -CheckOnly     只检查、只报告，不修改任何文件
  -SkipRocm      跳过第 2 步（不碰 ROCm 设备包）
  -SkipSelfTest  跳过第 4 步（不加载模型）
  -NoPause       结束时不停留
#>
param(
    [switch]$CheckOnly,
    [switch]$SkipRocm,
    [switch]$SkipSelfTest,
    [switch]$NoPause
)

$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

$pkg     = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv    = Join-Path $pkg 'venv'
$py      = Join-Path $venv 'Scripts\python.exe'
$app     = Join-Path $pkg 'app'
$epDir   = Join-Path $app 'model_weights\migraphx-ep'
$provider = Join-Path $epDir 'onnxruntime_providers_migraphx.dll'

$script:issues   = @()
$script:okCount  = 0

function Head($text) {
    Write-Host ''
    Write-Host ('== ' + $text + ' ' + ('-' * [Math]::Max(2, 58 - $text.Length)))
}
function Ok($text)   { Write-Host ('  [OK]   ' + $text) -ForegroundColor Green;  $script:okCount++ }
function Info($text) { Write-Host ('         ' + $text) }
function Warn($text) { Write-Host ('  [注意] ' + $text) -ForegroundColor Yellow; $script:issues += $text }
function Fail($text) { Write-Host ('  [失败] ' + $text) -ForegroundColor Red;    $script:issues += $text }

Write-Host '============================================================'
Write-Host ' Jasna 便携版 - 首次运行一键配置'
Write-Host '============================================================'
Info ('目录 : ' + $pkg)
Info ('参数 : ' + $(if ($CheckOnly) { 'CheckOnly（只检查，不做修改）' } else { '正常配置' }))

# ---------------------------------------------------------------------------
Head '0. 目录完整性'
# ---------------------------------------------------------------------------
if (-not (Test-Path $py))  { Fail ('找不到 venv\Scripts\python.exe：' + $py); }
else { Ok 'venv 正常' }
if (-not (Test-Path (Join-Path $app 'jasna\__main__.py'))) { Fail ('找不到 app\jasna：' + $app) } else { Ok 'app\jasna 正常' }
if (-not (Test-Path (Join-Path $pkg 'Start-Jasna.bat'))) { Warn '找不到 Start-Jasna.bat（启动器）' } else { Ok '启动器存在' }

if ($issues.Count -gt 0) {
    Write-Host ''
    Write-Host ' 目录不完整，后面的步骤没有意义，已停止。' -ForegroundColor Red
    Write-Host ' 请重新解压完整包（并确保解压到不含中文和 & 的路径）。' -ForegroundColor Red
    if (-not $NoPause) { Write-Host ''; Write-Host '按回车键退出...'; [void](Read-Host) }
    exit 1
}

# ---------------------------------------------------------------------------
Head '1. 路径检查'
# ---------------------------------------------------------------------------
if ($pkg -match '[^\x00-\x7F]') {
    Warn '当前路径含中文，启动器/脚本可能出问题，建议改到 D:\Jasna-AMD-Windows 这类纯英文路径'
} elseif ($pkg -match '&') {
    Warn '当前路径含 & 符号，批处理会解析错误，建议换一个不含 & 的路径'
} else {
    Ok ('路径可用：' + $pkg)
}

# ---------------------------------------------------------------------------
Head '2. Python 虚拟环境'
# ---------------------------------------------------------------------------
if (-not $CheckOnly) {
    $pyvenv = Join-Path $venv 'pyvenv.cfg'
    $cfg = @(
        ('home = ' + (Join-Path $pkg 'python312')),
        'include-system-site-packages = false',
        'version = 3.12.10',
        ('executable = ' + (Join-Path $pkg 'python312\python.exe')),
        ('base-prefix = ' + (Join-Path $pkg 'python312')),
        ('base-exec-prefix = ' + (Join-Path $pkg 'python312')),
        ('base-executable = ' + (Join-Path $pkg 'python312\python.exe'))
    )
    Set-Content -Path $pyvenv -Value $cfg -Encoding ascii
    $pth = Join-Path $venv 'Lib\site-packages\zz_jasna_path.pth'
    Set-Content -Path $pth -Value $app -Encoding ascii
    Ok 'venv 配置已写入（pyvenv.cfg + app 路径）'
} else {
    Info 'CheckOnly：跳过 venv 配置写入'
}
$env:PYTHONPATH = $app

$ver = & $py -c "import sys; print(sys.version.split()[0])" 2>$null | Select-Object -Last 1
if ($ver) { Ok ('Python ' + $ver) } else { Warn 'venv 里的 python 无法运行' }

# ---------------------------------------------------------------------------
Head '3. 显卡与架构'
# ---------------------------------------------------------------------------
$gfx = ''
$hipInfo = Join-Path $venv 'Scripts\hipInfo.exe'
if (Test-Path $hipInfo) {
    $out = & $hipInfo 2>$null | Out-String
    if ($out -match 'gcnArchName[:\s]+(gfx[0-9a-f]+)') { $gfx = $Matches[1] }
}
$gpuName = ''
$vids = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -match 'Radeon|AMD' } |
          Select-Object -ExpandProperty Name)
if ($vids.Count -gt 0) { $gpuName = ($vids -join ' + ') }
if (-not $gfx -and $gpuName) {
    $table = @(
        @('RX 7900 XTX|RX 7900 XT|RX 7900 GRE|W7900',   'gfx1100'),
        @('RX 7800 XT|RX 7700 XT|W7800',               'gfx1101'),
        @('RX 7600',                                   'gfx1102'),
        @('RX 9070|RX 9070 GRE',                       'gfx1201'),
        @('RX 9060',                                   'gfx1200'),
        @('Strix Halo|Radeon 80[0-9]S|Radeon 8050S',   'gfx1151'),
        @('RX 6900|RX 6800',                           'gfx1030'),
        @('RX 6700',                                   'gfx1031'),
        @('RX 6600',                                   'gfx1032')
    )
    foreach ($row in $table) { if ($gpuName -match $row[0]) { $gfx = $row[1]; break } }
}
if ($gpuName) { Ok ('显卡：' + $gpuName) } else { Warn '没有检测到 AMD 显卡' }
if ($gfx) { Ok ('架构：' + $gfx) } else { Warn '无法确定架构（可加 -Gfx 手动指定给 setup-rocm101-gpu.ps1）' }

$migxArchs = @('gfx1100', 'gfx1101', 'gfx1102', 'gfx1103', 'gfx1150', 'gfx1151', 'gfx1152', 'gfx1200', 'gfx1201')
if ($gfx) {
    if ($migxArchs -contains $gfx) {
        Ok '该架构在 MIGraphX 检测引擎支持范围内（RDNA3 / RDNA3.5 / RDNA4）'
    } else {
        Warn ('该架构不在 MIGraphX 检测引擎支持范围内（' + ($migxArchs -join ', ') + '）；' +
              '检测会自动走 PyTorch 路径，出片不受影响')
    }
}

# ---------------------------------------------------------------------------
Head '4. ROCm 内核包与注意力镜像'
# ---------------------------------------------------------------------------
if ($SkipRocm) {
    Info '已按参数跳过'
} elseif (-not $gfx) {
    Warn '架构未知，跳过内核包检查'
} else {
    $sp    = Join-Path $venv 'Lib\site-packages'
    $kpack = Join-Path $sp ('torch\.kpack\torch_' + $gfx + '.kpack')
    $images = Get-ChildItem (Join-Path $sp 'torch\lib\aotriton.images') -Recurse -File -ErrorAction SilentlyContinue
    $imgMb = 0
    if ($images) { $imgMb = [math]::Round((($images | Measure-Object Length -Sum).Sum) / 1MB, 1) }

    if (Test-Path $kpack) { Ok ('内核包已安装：torch_' + $gfx + '.kpack') } else { Warn ('缺少内核包 torch_' + $gfx + '.kpack（不装会报 hipErrorInvalidKernelFile）') }
    if ($images) { Ok ('注意力镜像：' + $images.Count + ' 个文件 / ' + $imgMb + ' MB（缺了会掉进慢 2 倍的 MATH 路径）') } else { Warn '缺少注意力镜像（能跑，但注意力会走慢路径）' }

    if ((Test-Path $kpack) -and $images) {
        Info '都已就绪，无需安装'
    } elseif ($CheckOnly) {
        Info 'CheckOnly：需要安装，但本次不执行'
        Info ('手动执行：setup-rocm101-gpu.ps1 -Gfx ' + $gfx + ' -WheelDir .\wheels')
    } else {
        $setup = Join-Path $pkg 'setup-rocm101-gpu.ps1'
        if (Test-Path $setup) {
            Info ('需要补装，正在执行 setup-rocm101-gpu.ps1 -Gfx ' + $gfx + ' ...')
            $wheelDir = Join-Path $pkg 'wheels'
            $setupArgs = @('-Gfx', $gfx)
            if (Test-Path $wheelDir) { $setupArgs += @('-WheelDir', $wheelDir); Info '（优先使用包内 wheels\ 离线安装）' }
            & $setup @setupArgs
            if ($LASTEXITCODE -eq 0) { Ok 'ROCm 组件安装流程结束' } else { Warn 'setup-rocm101-gpu.ps1 返回非 0，请看上面的输出' }
        } else {
            Warn '找不到 setup-rocm101-gpu.ps1，无法自动补装'
        }
    }
}

# ---------------------------------------------------------------------------
Head '5. MIGraphX 检测引擎运行库'
# ---------------------------------------------------------------------------
if (Test-Path $provider) {
    $files = Get-ChildItem $epDir -File
    $mb = [math]::Round((($files | Measure-Object Length -Sum).Sum) / 1MB, 1)
    Ok ('使用包内自带运行库：migraphx-ep\（' + $files.Count + ' 个文件 / ' + $mb + ' MB）')
    Info '不依赖 C:\Program Files\WindowsApps，也不依赖微软商店'
} else {
    Info '包内没有 migraphx-ep\，尝试从 Windows ML 已安装的组件复制一份进来'
    if ($CheckOnly) {
        Info 'CheckOnly：跳过复制'
    } else {
        $copy = @'
import json, os, shutil, sys
# run from the app folder: model_weights is resolved relative to the working
# directory, exactly like Start-Jasna.bat does (cd /d "%PKG%app")
dest = os.path.join("model_weights", "migraphx-ep")
info = {}
try:
    from jasna.mosaic.rfdetr_migraphx_runner import find_local_ep_dir, _ep_package
    if find_local_ep_dir() is not None:
        info["result"] = "already"
    else:
        src, prov = _ep_package()
        src = str(src)
        os.makedirs(dest, exist_ok=True)
        n = 0
        for name in os.listdir(src):
            p = os.path.join(src, name)
            if os.path.isfile(p) and (name.lower().endswith(".dll") or name.lower().endswith(".exe")):
                shutil.copy2(p, os.path.join(dest, name))
                n += 1
        info["result"] = "copied"
        info["count"] = n
        info["from"] = src
except Exception as exc:
    info["result"] = "error"
    info["error"] = repr(exc)[:300]
print("JASNA_EPCOPY " + json.dumps(info, ensure_ascii=False))
'@
        Push-Location $app
        $out = $copy | & $py - 2>$null | Select-String 'JASNA_EPCOPY' | Select-Object -Last 1
        Pop-Location
        try { $res = ($out.Line -replace '^.*JASNA_EPCOPY ', '') | ConvertFrom-Json } catch { $res = $null }
        if ($res -and $res.result -eq 'copied') {
            Ok ('已复制 ' + $res.count + ' 个运行库文件到 app\model_weights\migraphx-ep\')
        } elseif ($res -and $res.result -eq 'error') {
            Warn ('复制失败：' + $res.error)
        }
        if (Test-Path $provider) {
            Ok '包内运行库就绪'
        } else {
            Info '未复制成功：运行时仍会自动使用 Windows ML 组件（需要联网/商店可用）'
            Info '若想离线自带，可手动把 Windows ML 的 ExecutionProvider\ 整个目录复制到 app\model_weights\migraphx-ep\'
        }
    }
}

# ---------------------------------------------------------------------------
Head '6. 检测引擎自检'
# ---------------------------------------------------------------------------
if ($SkipSelfTest) {
    Info '已按参数跳过'
} elseif ($CheckOnly) {
    Info 'CheckOnly：跳过（自检会向 venv 写入 provider 并加载模型）'
} else {
    $selftest = @'
import glob, json, os
info = {}
try:
    import torch
    info["torch"] = torch.__version__
    info["gpu"] = torch.cuda.get_device_name(0)
    info["arch"] = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":")[0]
except Exception as exc:
    info["torch_error"] = repr(exc)[:200]
try:
    from jasna.mosaic import rfdetr_migraphx_runner as m
    info["ep_dir"] = str(m.find_local_ep_dir())
    info["arch_supported"] = bool(m.hip_arch() and m.hip_arch() in m.MIGRAPHX_SUPPORTED_ARCHS)
    m._prepare_ort_for_migraphx()
    onnx = sorted(glob.glob(os.path.join("model_weights", "*.migraphx.*.onnx")))
    if onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            onnx[0], sess_options=ort.SessionOptions(),
            providers=[("MIGraphXExecutionProvider",
                        {"device_id": "0", "migraphx_model_cache_dir": str(m.migraphx_cache_dir())}),
                       "CPUExecutionProvider"],
        )
        info["providers"] = sess.get_providers()
        info["onnx"] = os.path.basename(onnx[0])
except Exception as exc:
    info["ep_error"] = repr(exc)[:300]
print("JASNA_SELFTEST " + json.dumps(info, ensure_ascii=False))
'@
    Push-Location $app
    $raw = $selftest | & $py - 2>$null | Select-String 'JASNA_SELFTEST' | Select-Object -Last 1
    Pop-Location
    try { $st = ($raw.Line -replace '^.*JASNA_SELFTEST ', '') | ConvertFrom-Json } catch { $st = $null }

    if (-not $st) {
        Warn '自检没有返回结果（可能是 python 输出被截断），可手动运行 Start-Jasna.bat 观察启动日志'
    } else {
        if ($st.torch) { Ok ('PyTorch ' + $st.torch + '  /  ' + $st.gpu + '  /  ' + $st.arch) }
        elseif ($st.torch_error) { Warn ('PyTorch 加载失败：' + $st.torch_error) }
        if ($st.ep_dir) { Ok ('检测引擎运行库位置：' + $st.ep_dir) }
        if ($st.providers) {
            if ($st.providers -contains 'MIGraphXExecutionProvider') {
                Ok ('检测引擎自检通过：' + ($st.providers -join ' + ') + '  （模型 ' + $st.onnx + '）')
            } else {
                Warn ('检测引擎没有启用 MIGraphX（当前：' + ($st.providers -join ', ') + '）')
            }
        } elseif ($st.ep_error) {
            Warn ('检测引擎自检失败：' + $st.ep_error)
        } else {
            Info '没找到 MIGraphX 的 ONNX 导出，跳过会话自检'
        }
    }
}

# ---------------------------------------------------------------------------
Head '汇总'
# ---------------------------------------------------------------------------
if ($script:issues.Count -eq 0) {
    Write-Host ('  全部 ' + $script:okCount + ' 项检查通过，可以直接使用了。') -ForegroundColor Green
} else {
    Write-Host ('  通过 ' + $script:okCount + ' 项，另有 ' + $script:issues.Count + ' 项需要留意：') -ForegroundColor Yellow
    foreach ($i in $script:issues) { Write-Host ('   - ' + $i) -ForegroundColor Yellow }
    Write-Host ''
    Write-Host '  多数“留意”不影响出片（例如慢路径、非 MIGraphX 架构）。' -ForegroundColor Yellow
    Write-Host '  拿不准就把上面的输出整段发给 AI 编程助手，它能带你排掉。' -ForegroundColor Yellow
}

Write-Host ''
Write-Host '  下一步：双击 Start-Jasna.bat 启动图形界面。' -ForegroundColor Cyan
if ($CheckOnly) { Write-Host '  （本次是 CheckOnly，没有做任何修改）' -ForegroundColor Cyan }

if (-not $NoPause) {
    Write-Host ''
    Write-Host '按回车键退出...'
    [void](Read-Host)
}
exit 0
