#Requires -Version 5.1
# Windows 入口：没有 .venv 就当场建好（调用 setup.ps1），然后启动 Web 控制台。
#   .\run.ps1 --port 7861
# 停止：在 WebUI 里停工具栏任务，或回到这个窗口按 Ctrl+C。
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "首次运行：正在创建产品环境 .venv（安装依赖需要几分钟）..."
    try {
        & (Join-Path $PSScriptRoot "setup.ps1")
    } catch {
        Write-Host "环境创建失败：$($_.Exception.Message)" -ForegroundColor Red
        Write-Host "若提示禁止运行脚本：powershell -ExecutionPolicy Bypass -File .\setup.ps1" -ForegroundColor Red
        exit 1
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $python)) {
        Write-Host "环境创建失败，先单独运行 .\setup.ps1 看原因。" -ForegroundColor Red
        exit 1
    }
}

Write-Host "启动 WebUI（Ctrl+C 停止；端口见 uvicorn 输出）"
& $python -m moss_lora.cli web @args
