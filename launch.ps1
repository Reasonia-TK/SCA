param([switch]$Cpu, [switch]$NoBrowser, [switch]$Build, [string]$DataDir)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appDataExplicit = $PSBoundParameters.ContainsKey('DataDir') -or -not [string]::IsNullOrWhiteSpace($env:MEMORY_HOLE_DATA)
if ([string]::IsNullOrWhiteSpace($DataDir)) { $DataDir = $env:MEMORY_HOLE_DATA }
if ([string]::IsNullOrWhiteSpace($DataDir)) { $DataDir = Join-Path $PSScriptRoot 'data' }
$appDataDir = [System.IO.Path]::GetFullPath($DataDir)
$env:MEMORY_HOLE_DATA = $appDataDir
$env:PYTHONUTF8 = '1'
$appUrl = 'http://127.0.0.1:8765'
$appRunning = $false
try {
    $appState = Invoke-RestMethod -Uri "$appUrl/api/system" -TimeoutSec 2
    $appRunning = $appState.app_version -eq '0.1.0'
} catch { }
if ($appRunning -and $appDataExplicit -and $appState.data_dir -ne $appDataDir) {
    throw "別の保存先でサーバーが起動中です。実行中の計算を停止し、そのサーバーを終了してから再実行してください。指定した保存先: $appDataDir"
}
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
    & "$PSScriptRoot/.venv/Scripts/python.exe" -m memory_hole.storage --check-path $appDataDir
    if ($LASTEXITCODE -ne 0) { throw "保存先を使用できません: $appDataDir。上のエラーを確認してください。" }
    $appWorker = Start-Process -FilePath "$PSScriptRoot/.venv/Scripts/python.exe" -ArgumentList '-m','memory_hole' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $appDataDir 'server.log') -RedirectStandardError (Join-Path $appDataDir 'server.error.log') -PassThru
    Set-Content -LiteralPath (Join-Path $appDataDir 'server.pid') -Value $appWorker.Id
    for ($appTry=0; $appTry -lt 60; $appTry++) {
        try {
            $appState = Invoke-RestMethod -Uri "$appUrl/api/system" -TimeoutSec 2
            $appRunning = $appState.app_version -eq '0.1.0'
            if ($appRunning) { break }
        } catch { }
        Start-Sleep -Milliseconds 250
    }
    if (-not $appRunning) { throw "起動できませんでした。$(Join-Path $appDataDir 'server.error.log')を確認してください。" }
}
Write-Host "Memory Hole Lab: $appUrl"
Write-Host "保存先: $($appState.data_dir)"
if (-not $NoBrowser) { Start-Process $appUrl }
