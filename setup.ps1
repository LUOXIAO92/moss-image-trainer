#Requires -Version 5.1
<#
.SYNOPSIS
    创建（或修复）产品环境 .venv 并安装依赖（Windows）。

.DESCRIPTION
    与 setup.sh 等价：挑一个 Python 3.11+，在仓库根目录建 .venv，按显卡驱动装对应
    CUDA 版本的 torch，再装 requirements.txt，最后报告 torch / CUDA / 显卡。
    Windows 上的 torch 默认轮子是 CPU 版，所以这里必须显式指定 CUDA 索引。

    FlashAttention-2 的预编译轮子只有 Linux 版本，默认跳过；训练会自动改用 padded
    批次 + SDPA（更慢，结果一致）。要再试一次装轮子加 -WithFlashAttention。

.EXAMPLE
    .\setup.ps1

.EXAMPLE
    .\setup.ps1 -TorchIndex cu130 -Force

.EXAMPLE
    .\setup.ps1 -DryRun
#>
[CmdletBinding()]
param(
    # 删掉现有 .venv 重新创建。
    [switch]$Force,
    # 只打印将执行的 pip 命令，不下载安装。
    [switch]$DryRun,
    # 指定解释器，默认用 py 启动器挑 3.14 到 3.11，再退回 PATH 上的 python。
    [string]$Python = $env:PYTHON,
    # cu126 / cu128 / cu130 / cpu，默认按 NVIDIA 驱动版本挑。
    [string]$TorchIndex = $env:MOSS_TORCH_INDEX,
    # 额外尝试安装 FlashAttention-2 预编译轮子。
    [switch]$WithFlashAttention
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Stop-Setup([string]$Message) {
    Write-Host $Message -ForegroundColor Red
    exit 1
}

# 跑外部命令，退出码非 0 就停（PowerShell 默认不把原生命令的失败当异常）。
function Invoke-Native([string[]]$Command) {
    $exe = $Command[0]
    $rest = @($Command | Select-Object -Skip 1)
    & $exe @rest
    if ($LASTEXITCODE -ne 0) {
        Stop-Setup ("命令失败（exit {0}）：{1}" -f $LASTEXITCODE, ($Command -join ' '))
    }
}

# 探测用：吞掉 stderr 且不让它变成 terminating error。
function Invoke-Quiet([string]$Exe, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        return (& $Exe @Arguments 2>$null)
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Test-Interpreter([string]$Exe, [string[]]$Prefix) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Prefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previous
    }
}

# -- 解释器 ----------------------------------------------------------------

$pyExe = $null
$pyPrefix = @()
if ($Python) {
    if (-not (Test-Interpreter $Python @())) {
        Stop-Setup "指定的解释器不可用或低于 3.11：$Python"
    }
    $pyExe = $Python
} else {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        foreach ($version in '3.14', '3.13', '3.12', '3.11') {
            if (Test-Interpreter 'py' @("-$version")) {
                $pyExe = 'py'
                $pyPrefix = @("-$version")
                break
            }
        }
    }
    if (-not $pyExe) {
        foreach ($name in 'python', 'python3') {
            $found = Get-Command $name -ErrorAction SilentlyContinue
            if ($found -and (Test-Interpreter $found.Source @())) {
                $pyExe = $found.Source
                break
            }
        }
    }
    if (-not $pyExe) {
        Stop-Setup "找不到 Python 3.11+。装好 Python 后重试，或用 -Python C:\path\to\python.exe 指定。"
    }
}

$banner = & $pyExe @pyPrefix -c "import sys; print(sys.version.split()[0] + '  ' + sys.executable)"
Write-Host "Python: $banner"

# -- .venv -----------------------------------------------------------------

$venv = Join-Path $PSScriptRoot '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
$venvReady = (Test-Path -LiteralPath $venvPython) -and (Test-Path -LiteralPath (Join-Path $venv 'pyvenv.cfg'))

if ($Force -and (Test-Path -LiteralPath $venv)) {
    Write-Host '删除现有 .venv ...'
    Remove-Item -LiteralPath $venv -Recurse -Force
    $venvReady = $false
}

if (-not $venvReady) {
    if (Test-Path -LiteralPath (Join-Path $venv 'bin\python3')) {
        Stop-Setup '.venv 是 Linux/macOS 建的虚拟环境（没有 Scripts\python.exe）。加 -Force 删掉重建。'
    }
    Write-Host '创建 .venv ...'
    Invoke-Native (@($pyExe) + $pyPrefix + @('-m', 'venv', $venv))
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Stop-Setup "venv 创建后仍找不到 $venvPython，检查上面的输出。"
    }
}

# -- torch 索引 ------------------------------------------------------------

function Get-DriverVersion {
    if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) { return $null }
    $line = Invoke-Quiet 'nvidia-smi.exe' @('--query-gpu=driver_version', '--format=csv,noheader')
    if (-not $line) { return $null }
    $first = @($line)[0]
    if ($first -match '^\s*(\d+\.\d+)') { return [version]$Matches[1] }
    return $null
}

$indexUrl = $null
if ($env:TORCH_INDEX_URL) {
    $indexUrl = $env:TORCH_INDEX_URL
    Write-Host "torch 从 $indexUrl 安装（TORCH_INDEX_URL）"
} elseif ($TorchIndex) {
    if ($TorchIndex -notmatch '^(cu\d+|cpu)$') {
        Stop-Setup "-TorchIndex 只接受 cu126 / cu128 / cu130 / cpu，收到：$TorchIndex"
    }
    $indexUrl = "https://download.pytorch.org/whl/$TorchIndex"
    Write-Host "torch 从 $indexUrl 安装（-TorchIndex）"
} else {
    $driver = Get-DriverVersion
    if (-not $driver) {
        $tag = 'cu128'
        Write-Host '读不到 NVIDIA 驱动版本（没装卡，或 nvidia-smi 不在 PATH）：按 cu128 装。'
        Write-Host '  没有 NVIDIA 显卡时用 -TorchIndex cpu。'
    } elseif ($driver -ge [version]'580.0') {
        $tag = 'cu130'
        Write-Host "驱动 $driver → torch cu130"
    } elseif ($driver -ge [version]'570.0') {
        $tag = 'cu128'
        Write-Host "驱动 $driver → torch cu128"
    } elseif ($driver -ge [version]'527.0') {
        $tag = 'cu126'
        Write-Host "驱动 $driver → torch cu126"
    } else {
        $tag = 'cu126'
        Write-Host "驱动 $driver 低于 CUDA 12.6 的要求，先更新显卡驱动；仍按 cu126 装。" -ForegroundColor Yellow
    }
    $indexUrl = "https://download.pytorch.org/whl/$tag"
}

# -- 安装 -------------------------------------------------------------------

$pip = @($venvPython, '-m', 'pip')
if ($DryRun) {
    Write-Host '（dry-run：以下命令不会执行）'
    Write-Host "  .venv\Scripts\python.exe -m pip install --upgrade pip"
    Write-Host "  .venv\Scripts\python.exe -m pip install torch torchvision --index-url $indexUrl"
    Write-Host "  .venv\Scripts\python.exe -m pip install -r requirements.txt"
    Write-Host "环境已就绪（dry-run）：$venv"
    exit 0
}

Invoke-Native ($pip + @('install', '--upgrade', 'pip'))
Write-Host "安装 torch + torchvision（$indexUrl）"
Invoke-Native ($pip + @('install', 'torch', 'torchvision', '--index-url', $indexUrl))
Write-Host '安装 requirements.txt'
Invoke-Native ($pip + @('install', '-r', 'requirements.txt'))

# -- 自检 -------------------------------------------------------------------

$probe = @'
import torch
ok = torch.cuda.is_available()
print('torch ' + torch.__version__ + (' / CUDA 可用' if ok else ' / CUDA 不可用'))
if ok:
    print('GPU: ' + torch.cuda.get_device_name(0))
'@
& $venvPython -c $probe
if ($LASTEXITCODE -ne 0) {
    Stop-Setup 'torch 导入失败。若是 CPU 版或驱动与 CUDA 版本不匹配，换索引重装：.\setup.ps1 -Force -TorchIndex cu126'
}

# -- FlashAttention-2 -------------------------------------------------------

if ($WithFlashAttention) {
    Write-Host ''
    Write-Host 'FlashAttention-2'
    & $venvPython (Join-Path $PSScriptRoot 'setup_fa2.py') install
    if ($LASTEXITCODE -ne 0) {
        Write-Host '警告：FlashAttention-2 未安装，打包训练退回 padded + SDPA（更慢，结果一致）。' -ForegroundColor Yellow
    }
} else {
    Write-Host 'FlashAttention-2：跳过。当前轮子源只有 Linux 版本，训练自动用 padded 批次 + SDPA；要试装加 -WithFlashAttention。'
}

Write-Host ''
Write-Host "环境已就绪：.\run.ps1 --port 7869"
