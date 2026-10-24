param([switch]$Notebook)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create the project virtual environment first. See README.md.'
}
Push-Location -LiteralPath $taskProjectRoot
try {
    & $taskPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed; fix them before recording.' }
    if ($Notebook) {
        & $taskPython -m jupyterlab demo/quickstart.ipynb
    } else {
        & $taskPython demo/integrated_quickstart.py
    }
    if ($LASTEXITCODE -ne 0) { throw 'Demo did not finish successfully.' }
} finally {
    Pop-Location
}
