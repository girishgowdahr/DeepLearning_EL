# ForestGuard AI — Artifacts Installation & Verification Script
param(
    [Parameter(Mandatory=$false)][string]$YoloZip = "",
    [Parameter(Mandatory=$false)][string]$CnnZip = ""
)

$ErrorActionPreference = "Stop"
$BaseDir = (Get-Item $PSScriptRoot).Parent.FullName
$ModelWeightsDir = Join-Path $BaseDir "model_weights"
$MetricsDir = Join-Path $BaseDir "results\metrics"
$GraphsDir = Join-Path $BaseDir "results\graphs"
$PredictionsDir = Join-Path $BaseDir "results\predictions"
$TempDir = Join-Path $BaseDir "run\temp_artifacts"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🌲 ForestGuard AI Artifact Installer" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Backup existing weights if present
$ExistingWeights = Get-ChildItem -Path $ModelWeightsDir -File -ErrorAction SilentlyContinue
if ($ExistingWeights.Count -gt 0) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BackupDir = Join-Path $ModelWeightsDir "backup_$Timestamp"
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    Write-Host "📦 Backing up existing weights to $BackupDir..." -ForegroundColor Yellow
    foreach ($f in $ExistingWeights) {
        Copy-Item -Path $f.FullName -Destination (Join-Path $BackupDir $f.Name)
    }
}

# Ensure destination folders exist
New-Item -ItemType Directory -Path $ModelWeightsDir -Force | Out-Null
New-Item -ItemType Directory -Path $MetricsDir -Force | Out-Null
New-Item -ItemType Directory -Path $GraphsDir -Force | Out-Null
New-Item -ItemType Directory -Path $PredictionsDir -Force | Out-Null

# Clean / create temp dir
if (Test-Path $TempDir) { Remove-Item -Path $TempDir -Recurse -Force }
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

function Process-Zip($ZipPath, $ModelName) {
    if (-not (Test-Path $ZipPath)) {
        Write-Host "⚠️ Warning: $ModelName zip not found at $ZipPath. Skipping." -ForegroundColor Yellow
        return
    }
    Write-Host "`n🗜️ Extracting $ModelName from $ZipPath..." -ForegroundColor Cyan
    $ExtractSub = Join-Path $TempDir $ModelName
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractSub -Force

    # Copy files based on extension / name
    Get-ChildItem -Path $ExtractSub -Recurse -File | ForEach-Object {
        $file = $_
        if ($file.Name -match "\.(pt|onnx)$" -or $file.Name -eq "MODEL_INFO.json") {
            Copy-Item -Path $file.FullName -Destination (Join-Path $ModelWeightsDir $file.Name) -Force
            Write-Host "  + Installed Weight/Config: $($file.Name)" -ForegroundColor Green
        }
        elseif ($file.Name -match "metrics\.json$") {
            Copy-Item -Path $file.FullName -Destination (Join-Path $MetricsDir $file.Name) -Force
            Write-Host "  + Installed Metrics: $($file.Name)" -ForegroundColor Green
        }
        elseif ($file.Name -match "\.(png|jpg|jpeg)$") {
            Copy-Item -Path $file.FullName -Destination (Join-Path $GraphsDir $file.Name) -Force
            Write-Host "  + Installed Plot: $($file.Name)" -ForegroundColor Green
        }
    }
}

if ($YoloZip) { Process-Zip -ZipPath $YoloZip -ModelName "YOLO" }
if ($CnnZip) { Process-Zip -ZipPath $CnnZip -ModelName "CNN" }

# Cleanup temp dir
if (Test-Path $TempDir) { Remove-Item -Path $TempDir -Recurse -Force }

# 2. Print Installed Manifest & SHA-256 Hashes
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "📋 INSTALLED WEIGHTS & CHECKSUMS (model_weights/)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

$InstalledFiles = Get-ChildItem -Path $ModelWeightsDir -File
if ($InstalledFiles.Count -eq 0) {
    Write-Host "No files found in model_weights/." -ForegroundColor Red
} else {
    foreach ($file in $InstalledFiles) {
        $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash
        $sizeMB = [math]::Round($file.Length / 1MB, 2)
        Write-Host ("{0,-25} {1,8} MB | SHA256: {2}" -f $file.Name, $sizeMB, $hash) -ForegroundColor White
    }
}

Write-Host "`n✅ Artifact installation complete. Run 'python scripts\validate_models.py' next." -ForegroundColor Green
