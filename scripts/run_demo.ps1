# ForestGuard AI — Demo Day Master Launcher
param (
    [switch]$SelfSigned = $false
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🌲 ForestGuard AI — Academic Prototype Demo Launcher" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$venvPython = ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not detected. Running setup first..." -ForegroundColor Yellow
    .\scripts\setup_windows.ps1
}

# 1. Ensure Frontend Build Exists
if (-not (Test-Path "frontend\dist\index.html")) {
    Write-Host "Building frontend distribution..." -ForegroundColor Yellow
    Push-Location frontend
    npm run build
    Pop-Location
}

# Ensure run directory exists
New-Item -ItemType Directory -Force -Path "run" | Out-Null
$tunnelFile = "run\tunnel_url.txt"
if (Test-Path $tunnelFile) { Remove-Item $tunnelFile }

# 2. Check for Cloudflare Tunnel
$cloudflared = Get-Command "cloudflared" -ErrorAction SilentlyContinue
$tunnelProcess = $null

if ($cloudflared -and -not $SelfSigned) {
    Write-Host "Starting Cloudflare Quick Tunnel for Phone Camera HTTPS..." -ForegroundColor Yellow
    $logFile = "run\cloudflared.log"
    $tunnelProcess = Start-Process -FilePath "cloudflared" -ArgumentList "tunnel --url http://localhost:8000" -RedirectStandardError $logFile -PassThru -WindowStyle Hidden
    
    # Wait for tunnel URL
    $foundUrl = $null
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $logFile) {
            $content = Get-Content $logFile -Raw
            if ($content -match '(https://[a-zA-Z0-9-]+\.trycloudflare\.com)') {
                $foundUrl = $matches[1]
                Set-Content -Path $tunnelFile -Value $foundUrl
                break
            }
        }
    }

    if ($foundUrl) {
        Write-Host "==========================================================" -ForegroundColor Green
        Write-Host "📱 PHONE CAMERA STREAM URL (HTTPS):" -ForegroundColor Green
        Write-Host "   $foundUrl/camera" -ForegroundColor White -BackgroundColor DarkGreen
        Write-Host "==========================================================" -ForegroundColor Green
    }
}

# 3. Detect LAN IP
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi*", "Ethernet*" -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -notlike "169.*" -and $_.IPAddress -notlike "127.*" } | Select-Object -First 1).IPAddress
if (-not $lanIp) { $lanIp = "127.0.0.1" }

Write-Host "Laptop Dashboard: http://localhost:8000" -ForegroundColor Cyan
Write-Host "LAN IP: $lanIp" -ForegroundColor Cyan

# 4. Start Uvicorn Server
try {
    if ($SelfSigned) {
        Write-Host "Starting with Self-Signed SSL (Local HTTPS)..." -ForegroundColor Yellow
        & $venvPython scripts\gen_selfsigned_cert.py
        & $venvPython -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile run\key.pem --ssl-certfile run\cert.pem
    } else {
        & $venvPython -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
    }
} finally {
    if ($tunnelProcess -and -not $tunnelProcess.HasExited) {
        Stop-Process -Id $tunnelProcess.Id -Force
    }
}
