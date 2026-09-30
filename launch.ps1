param([switch]$Cpu, [switch]$NoBrowser, [switch]$Build)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appUrl = 'http://127.0.0.1:8765'
$appRunning = $false
try {
    $appState = Invoke-RestMethod -Uri "$appUrl/api/system" -TimeoutSec 2
    $appRunning = $appState.app_version -eq '0.1.0'
} catch { }
if (-not $appRunning) {
    if ($Cpu) { & uv sync } else { & uv sync --extra gpu }
    if ($LASTEXITCODE -ne 0) { throw 'Python環境の同期に失敗しました。' }
    if ($Build -or -not (Test-Path -LiteralPath 'frontend/dist/index.html')) {
        Push-Location -LiteralPath 'frontend'
        try {
            if (-not (Test-Path -LiteralPath 'node_modules')) { & npm.cmd ci }
            if ($LASTEXITCODE -ne 0) { throw '画面の依存関係導入に失敗しました。' }
            & npm.cmd run build
            if ($LASTEXITCODE -ne 0) { throw '画面ビルドに失敗しました。' }
        } finally { Pop-Location }
    }
    New-Item -ItemType Directory -Path 'data' -Force | Out-Null
    $appWorker = Start-Process -FilePath "$PSScriptRoot/.venv/Scripts/python.exe" -ArgumentList '-m','memory_hole' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput "$PSScriptRoot/data/server.log" -RedirectStandardError "$PSScriptRoot/data/server.error.log" -PassThru
    Set-Content -LiteralPath 'data/server.pid' -Value $appWorker.Id
    for ($appTry=0; $appTry -lt 60; $appTry++) {
        try {
            $appState = Invoke-RestMethod -Uri "$appUrl/api/system" -TimeoutSec 2
            $appRunning = $appState.app_version -eq '0.1.0'
            if ($appRunning) { break }
        } catch { }
        Start-Sleep -Milliseconds 250
    }
    if (-not $appRunning) { throw '起動できませんでした。data/server.error.logを確認してください。' }
}
Write-Host "Memory Hole Lab: $appUrl"
if (-not $NoBrowser) { Start-Process $appUrl }
