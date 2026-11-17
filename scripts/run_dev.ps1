# ForestGuard AI — Development Server Launcher (Reload enabled)
Write-Host "🌲 Starting ForestGuard AI in Development Mode..." -ForegroundColor Cyan

$venvPython = ".venv\Scripts\python.exe"

# Start FastAPI backend in background job
$backendJob = Start-Job -ScriptBlock {
    param($python)
    & $python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
} -ArgumentList $venvPython

Write-Host "Backend started at http://localhost:8000" -ForegroundColor Green
Write-Host "Starting Vite frontend dev server at http://localhost:5173..." -ForegroundColor Yellow

Push-Location frontend
try {
    npm run dev
} finally {
    Pop-Location
    Stop-Job $backendJob
    Remove-Job $backendJob
}
