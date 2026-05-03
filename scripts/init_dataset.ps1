# Creates dataset/cheating and dataset/not_cheating with sample1.jpg + sample2.jpg
# by calling the Python helper (same bytes as write_placeholder_jpegs.py).
# From project root:
#   powershell -ExecutionPolicy Bypass -File scripts/init_dataset.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$venvPy = Join-Path $root "venv\Scripts\python.exe"
$python = $null
if (Test-Path $venvPy) {
    $python = $venvPy
} else {
    foreach ($name in @("python", "py", "python3")) {
        try {
            $c = Get-Command $name -ErrorAction Stop
            $python = $c.Source
            break
        } catch { }
    }
}

if (-not $python) {
    Write-Host "Python was not found. Create a venv, install deps, or run:"
    Write-Host "  python scripts/write_placeholder_jpegs.py"
    exit 1
}

& $python "scripts/write_placeholder_jpegs.py"
exit $LASTEXITCODE
