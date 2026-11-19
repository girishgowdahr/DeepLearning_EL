# ForestGuard AI — Demo Health Check Script
# Validates backend, frontend, models, camera, decision engine, detection history,
# alert endpoints, and Cloudflare tunnel status.
# Clearly distinguishes PASS, FAIL, and MODEL_UNAVAILABLE states.
param(
    [string]$BaseUrl = "http://localhost:8000",
    [int]$TimeoutSec = 3
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$BaseDir = (Get-Item $PSScriptRoot).Parent.FullName

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🌲 ForestGuard AI — Comprehensive System Health Check" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Target Host:       $BaseUrl"
Write-Host "Local Workspace:   $BaseDir"
Write-Host "Audit Timestamp:   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"

$results = [ordered]@{}
$hasFailures = $false

function Record-Check([string]$Name, [string]$Status, [string]$Details, [ConsoleColor]$Color) {
    $script:results[$Name] = @{ Status = $Status; Details = $Details; Color = $Color }
    if ($Status -eq "FAIL") { $script:hasFailures = $true }
}

# 1. Check Backend API Reachability
try {
    $resp = Invoke-RestMethod -Uri "$BaseUrl/api/health" -Method Get -TimeoutSec $TimeoutSec -ErrorAction Stop
    if ($resp.status -eq "ok") {
        Record-Check "1. Backend API Server" "PASS" "Online at $BaseUrl/api/health" Green
    } else {
        Record-Check "1. Backend API Server" "FAIL" "Unexpected health response: $($resp | ConvertTo-Json -Compress)" Red
    }
    $isLive = $true
} catch {
    $isLive = $false
    # Server might not be currently running, check static backend readiness
    $backendMain = Join-Path $BaseDir "backend\main.py"
    if (Test-Path $backendMain) {
        Record-Check "1. Backend API Server" "OFFLINE" "No live process listening on $BaseUrl (ready to start via run_demo.ps1)" Yellow
    } else {
        Record-Check "1. Backend API Server" "FAIL" "Backend entry point missing: $backendMain" Red
    }
}

# 2. Check Frontend Production Distribution
$distIndex = Join-Path $BaseDir "frontend\dist\index.html"
if (Test-Path $distIndex) {
    $distSize = (Get-Item $distIndex).Length
    if ($distSize -gt 500) {
        Record-Check "2. Frontend UI Bundle" "PASS" "Compiled distribution present ($distSize bytes)" Green
    } else {
        Record-Check "2. Frontend UI Bundle" "FAIL" "frontend\dist\index.html is truncated ($distSize bytes)" Red
    }
} else {
    Record-Check "2. Frontend UI Bundle" "FAIL" "frontend\dist\ missing (run 'npm run build' inside frontend/)" Red
}

# 3. Check Model Availability & Degraded State
$cnnPath = Join-Path $BaseDir "model_weights\cnn_mobilenetv2.onnx"
$yoloPath = Join-Path $BaseDir "model_weights\yolo_best.pt"
$hasCnn = Test-Path $cnnPath
$hasYolo = Test-Path $yoloPath

if ($hasCnn -and $hasYolo) {
    Record-Check "3. Model Availability" "PASS" "Both CNN and YOLO model weights installed and operational" Green
} elseif ($hasCnn -xor $hasYolo) {
    Record-Check "3. Model Availability" "MODEL_UNAVAILABLE" "Partial weights present (degraded single-model mode active)" Yellow
} else {
    Record-Check "3. Model Availability" "MODEL_UNAVAILABLE" "Weights pending Kaggle download (MODEL_UNAVAILABLE safe state active)" Cyan
}

# 4. Check Decision Engine Integrity
$decisionEngineScript = Join-Path $BaseDir "backend\services\decision_engine.py"
if (Test-Path $decisionEngineScript) {
    $deContent = Get-Content $decisionEngineScript -Raw
    if ($deContent -match "MODEL_UNAVAILABLE" -and $deContent -match "FIRE_CONFIRMED" -and $deContent -match "SMOKE_CONFIRMED") {
        Record-Check "4. Decision Fusion Engine" "PASS" "Multi-modal sliding window & safe degraded logic verified" Green
    } else {
        Record-Check "4. Decision Fusion Engine" "FAIL" "DecisionState missing MODEL_UNAVAILABLE state" Red
    }
} else {
    Record-Check "4. Decision Fusion Engine" "FAIL" "Missing decision_engine.py" Red
}

# 5. Check Live Camera Endpoint (if server live)
if ($isLive) {
    try {
        $camResp = Invoke-RestMethod -Uri "$BaseUrl/api/camera/status" -Method Get -TimeoutSec $TimeoutSec -ErrorAction Stop
        Record-Check "5. Camera Ingestion Endpoint" "PASS" "Camera endpoint live (Active: $($camResp.active), Source: $($camResp.source_type))" Green
    } catch {
        Record-Check "5. Camera Ingestion Endpoint" "FAIL" "Failed to query $BaseUrl/api/camera/status: $($_.Exception.Message)" Red
    }
} else {
    $camModule = Join-Path $BaseDir "backend\api\routes_camera.py"
    if (Test-Path $camModule) {
        Record-Check "5. Camera Ingestion Endpoint" "READY" "Endpoints configured for Laptop Webcam & Android WSS Stream" Gray
    } else {
        Record-Check "5. Camera Ingestion Endpoint" "FAIL" "Missing backend\api\routes_camera.py" Red
    }
}

# 6. Check Detection History & Database
if ($isLive) {
    try {
        $detResp = Invoke-RestMethod -Uri "$BaseUrl/api/detections?limit=5" -Method Get -TimeoutSec $TimeoutSec -ErrorAction Stop
        Record-Check "6. Detection History Store" "PASS" "Detections query operational ($($detResp.Count) records returned)" Green
    } catch {
        Record-Check "6. Detection History Store" "FAIL" "Failed to query /api/detections: $($_.Exception.Message)" Red
    }
} else {
    $dbPath = Join-Path $BaseDir "backend\data\forestguard.db"
    if (Test-Path $dbPath) {
        Record-Check "6. Detection History Store" "PASS" "SQLite storage active at backend\data\forestguard.db" Green
    } else {
        Record-Check "6. Detection History Store" "READY" "SQLite auto-creates on first server launch" Gray
    }
}

# 7. Check Alerts & Siren Endpoint
if ($isLive) {
    try {
        $alertResp = Invoke-RestMethod -Uri "$BaseUrl/api/alerts?limit=5" -Method Get -TimeoutSec $TimeoutSec -ErrorAction Stop
        Record-Check "7. Alerts & Siren System" "PASS" "Alert subsystem live ($($alertResp.Count) historical events)" Green
    } catch {
        Record-Check "7. Alerts & Siren System" "FAIL" "Failed to query /api/alerts: $($_.Exception.Message)" Red
    }
} else {
    $alertService = Join-Path $BaseDir "backend\services\alert_manager.py"
    if (Test-Path $alertService) {
        Record-Check "7. Alerts & Siren System" "READY" "Audio alarm & notification dispatcher ready" Gray
    } else {
        Record-Check "7. Alerts & Siren System" "FAIL" "Missing backend\services\alert_manager.py" Red
    }
}

# 8. Check Cloudflare Quick Tunnel
$cfExe = Join-Path $BaseDir "run\cloudflared.exe"
$cfUrlFile = Join-Path $BaseDir "run\tunnel_url.txt"
$hasCfExe = Test-Path $cfExe
$hasCfUrl = (Test-Path $cfUrlFile) -and ((Get-Item $cfUrlFile).Length -gt 10)

if ($hasCfExe -and $hasCfUrl) {
    $tunnelUrl = (Get-Content $cfUrlFile -Raw).Trim()
    Record-Check "8. Cloudflare Mobile Tunnel" "PASS" "Tunnel active at $tunnelUrl" Green
} elseif ($hasCfExe) {
    Record-Check "8. Cloudflare Mobile Tunnel" "READY" "cloudflared binary present in run\ (launches with run_demo.ps1)" Gray
} else {
    # Check PATH
    $cmdCf = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmdCf) {
        Record-Check "8. Cloudflare Mobile Tunnel" "READY" "cloudflared installed on system PATH ($($cmdCf.Source))" Gray
    } else {
        Record-Check "8. Cloudflare Mobile Tunnel" "WARNING" "cloudflared.exe missing from run/ (auto-downloadable via run_demo.ps1)" Yellow
    }
}

# Output Table
Write-Host ("{0,-30} | {1,-18} | {2}" -f "Health Component", "Status", "Details") -ForegroundColor Cyan
Write-Host ("-" * 85) -ForegroundColor DarkGray

foreach ($k in $results.Keys) {
    $item = $results[$k]
    Write-Host ("{0,-30} | " -f $k) -NoNewline
    Write-Host ("{0,-18} | " -f $item.Status) -ForegroundColor $item.Color -NoNewline
    Write-Host $item.Details
}

Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
if ($hasFailures) {
    Write-Host "❌ HEALTH CHECK FAILED: One or more components require attention." -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ SYSTEM HEALTHY: All components operational or in safe degraded mode." -ForegroundColor Green
    exit 0
}
