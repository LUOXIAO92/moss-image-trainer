$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
  Write-Error "未找到 .venv。请先创建环境（Windows: python -m venv .venv；WSL: ./setup.sh）。"
  exit 1
}
& $python -m moss_lora.cli web @args
