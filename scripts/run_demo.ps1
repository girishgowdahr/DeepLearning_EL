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

# Per-run phone access token (192-bit, URL-safe). The backend reads it from FG_PHONE_TOKEN.
# It is never written to git, never placed in a query string (only in the URL #fragment,
# which browsers do not send to servers), and a new one is generated on every run.
$bytes = New-Object byte[] 24
[System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$env:FG_PHONE_TOKEN = [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
Remove-Item "run\phone_token.txt" -ErrorAction SilentlyContinue

# 2. Check for Cloudflare Tunnel
$cloudflaredCmd = "cloudflared"
$cloudflared = Get-Command "cloudflared" -ErrorAction SilentlyContinue
if (-not $cloudflared) {
    $localBinary = Join-Path (Get-Location) "run\cloudflared.exe"
    if (Test-Path $localBinary) {
        $cloudflared = Get-Item $localBinary
        $cloudflaredCmd = $localBinary
    }
}
$tunnelProcess = $null

if ($cloudflared -and -not $SelfSigned) {
    Write-Host "Starting Cloudflare Quick Tunnel for Phone Camera HTTPS..." -ForegroundColor Yellow
    $logFile = "run\cloudflared.log"
    $tunnelProcess = Start-Process -FilePath $cloudflaredCmd -ArgumentList "tunnel --url http://localhost:8000" -RedirectStandardError $logFile -RedirectStandardOutput $logFile -PassThru -WindowStyle Hidden
    
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
        Write-Host "   $foundUrl/camera#t=$($env:FG_PHONE_TOKEN)" -ForegroundColor White -BackgroundColor DarkGreen
        Write-Host "   (contains a secret access token - open it only on YOUR phone;" -ForegroundColor Yellow
        Write-Host "    easiest: scan the QR on the dashboard Live Camera page)" -ForegroundColor Yellow
        Write-Host "==========================================================" -ForegroundColor Green
    } else {
        Write-Host "Tunnel URL not found (blocked network?). Fallbacks:" -ForegroundColor Red
        Write-Host "  1) Use your phone's hotspot for the laptop and re-run this script" -ForegroundColor Red
        Write-Host "  2) .\scripts\run_demo.ps1 -SelfSigned   (same Wi-Fi, accept the certificate warning once)" -ForegroundColor Red
        Write-Host "  3) Install the 'IP Webcam' Android app and use Live Camera > IP camera" -ForegroundColor Red
    }
} elseif (-not $SelfSigned) {
    Write-Host "cloudflared not installed: phone camera needs HTTPS. Run: winget install Cloudflare.cloudflared" -ForegroundColor Red
    Write-Host "or use .\scripts\run_demo.ps1 -SelfSigned" -ForegroundColor Red
}

# 3. Detect LAN IP
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi*", "Ethernet*" -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -notlike "169.*" -and $_.IPAddress -notlike "127.*" } | Select-Object -First 1).IPAddress
if (-not $lanIp) { $lanIp = "127.0.0.1" }

Write-Host "Laptop Dashboard: http://localhost:8000  (laptop only - the tunnel cannot open it)" -ForegroundColor Cyan
Write-Host "LAN IP: $lanIp" -ForegroundColor Cyan
Write-Host "Stop with Ctrl+C. The tunnel URL is temporary and dies when this window closes." -ForegroundColor DarkGray

# 4. Start Uvicorn Server
try {
    if ($SelfSigned) {
        Write-Host "Starting with Self-Signed SSL (Local HTTPS)..." -ForegroundColor Yellow
        Write-Host "==========================================================" -ForegroundColor Green
        Write-Host "📱 PHONE CAMERA URL (same Wi-Fi; tap Advanced > Proceed once):" -ForegroundColor Green
        Write-Host "   https://${lanIp}:8000/camera#t=$($env:FG_PHONE_TOKEN)" -ForegroundColor White -BackgroundColor DarkGreen
        Write-Host "   Laptop dashboard in this mode: https://localhost:8000" -ForegroundColor Cyan
        Write-Host "==========================================================" -ForegroundColor Green
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
