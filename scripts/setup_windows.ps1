# ForestGuard AI — Windows PowerShell Setup Script
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🌲 ForestGuard AI — Environment Setup (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Locate Python 3.11
$pythonExe = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python.exe"
}

Write-Host "[1/4] Creating virtual environment (.venv) using $pythonExe..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    & $pythonExe -m venv .venv
}

$venvPython = ".venv\Scripts\python.exe"

# 2. Upgrade pip
Write-Host "[2/4] Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip

# 3. Install CPU PyTorch & Backend Dependencies
Write-Host "[3/4] Installing CPU PyTorch & backend requirements..." -ForegroundColor Yellow
& $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $venvPython -m pip install -r backend\requirements.txt

# 4. Frontend Setup
Write-Host "[4/4] Installing frontend dependencies & building distribution..." -ForegroundColor Yellow
Push-Location frontend
npm install
npm run build
Pop-Location

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "✅ ForestGuard AI Setup Complete!" -ForegroundColor Green
Write-Host "Run .\scripts\run_demo.ps1 to start the prototype system." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
