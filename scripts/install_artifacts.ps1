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
Write-Host "🌲 ForestGuard AI Artifact Installer & Checksum Verifier" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# Check if arguments provided; if not, look in common download spots
if (-not $YoloZip -and -not $CnnZip) {
    $candYolo = Join-Path $BaseDir "yolo_artifacts.zip"
    $candCnn = Join-Path $BaseDir "cnn_artifacts.zip"
    if (Test-Path $candYolo) { $YoloZip = $candYolo }
    if (Test-Path $candCnn) { $CnnZip = $candCnn }
    
    if (-not $YoloZip -and -not $CnnZip) {
        Write-Host "Usage: .\scripts\install_artifacts.ps1 -YoloZip <path> -CnnZip <path>" -ForegroundColor Yellow
        Write-Host "No zip files specified, and no default artifacts found in project root." -ForegroundColor DarkGray
        exit 0
    }
}

# 1. Backup existing weights if present
$ExistingWeights = Get-ChildItem -Path $ModelWeightsDir -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -match "\.(pt|onnx)$" }
if ($ExistingWeights.Count -gt 0) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BackupDir = Join-Path $ModelWeightsDir "backup_$Timestamp"
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    Write-Host "📦 Backing up existing weights to $BackupDir..." -ForegroundColor Yellow
    foreach ($f in $ExistingWeights) {
        Copy-Item -Path $f.FullName -Destination (Join-Path $BackupDir $f.Name) -Force
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

function Process-Zip($ZipPath, $ModelName, $RequiredWeightRegex) {
    if (-not (Test-Path $ZipPath)) {
        Write-Host "⚠️ Warning: $ModelName zip not found at '$ZipPath'. Skipping." -ForegroundColor Yellow
        return $false
    }

    Write-Host "`n🗜️ Extracting $ModelName from $ZipPath..." -ForegroundColor Cyan
    $ExtractSub = Join-Path $TempDir $ModelName
    try {
        Expand-Archive -Path $ZipPath -DestinationPath $ExtractSub -Force
    } catch {
        Write-Host "❌ Error: Failed to extract $ModelName archive! The zip file may be corrupted." -ForegroundColor Red
        Write-Host "   Details: $($_.Exception.Message)" -ForegroundColor DarkRed
        return $false
    }

    $extractedFiles = Get-ChildItem -Path $ExtractSub -Recurse -File
    if ($extractedFiles.Count -eq 0) {
        Write-Host "❌ Error: $ModelName archive is empty!" -ForegroundColor Red
        return $false
    }

    # Verify presence of essential model weight
    $hasWeight = $false
    foreach ($file in $extractedFiles) {
        if ($file.Name -match $RequiredWeightRegex) {
            if ($file.Length -lt 100KB) {
                Write-Host "⚠️ Warning: Extracted weight $($file.Name) is suspiciously small ($($file.Length) bytes)." -ForegroundColor Yellow
            }
            $hasWeight = $true
        }
    }

    if (-not $hasWeight) {
        Write-Host "⚠️ Warning: $ModelName archive does not contain expected model weights ($RequiredWeightRegex)!" -ForegroundColor Yellow
    }

    # Copy files based on destination mapping
    foreach ($file in $extractedFiles) {
        if ($file.Name -match "\.(pt|onnx)$" -or $file.Name -eq "MODEL_INFO.json" -or $file.Name -eq "MANIFEST.txt") {
            Copy-Item -Path $file.FullName -Destination (Join-Path $ModelWeightsDir $file.Name) -Force
            Write-Host "  + Installed Weight/Config: $($file.Name)" -ForegroundColor Green
        }
        elseif ($file.Name -match "metrics\.json$") {
            Copy-Item -Path $file.FullName -Destination (Join-Path $MetricsDir $file.Name) -Force
            Write-Host "  + Installed Metrics:       $($file.Name)" -ForegroundColor Green
        }
        elseif ($file.Name -match "\.(png|jpg|jpeg)$") {
            Copy-Item -Path $file.FullName -Destination (Join-Path $GraphsDir $file.Name) -Force
            Write-Host "  + Installed Plot:          $($file.Name)" -ForegroundColor Green
        }
    }
    return $true
}

$yoloSuccess = $true
$cnnSuccess = $true

if ($YoloZip) {
    $yoloSuccess = Process-Zip -ZipPath $YoloZip -ModelName "YOLO" -RequiredWeightRegex "yolo_best\.(pt|onnx)"
}
if ($CnnZip) {
    $cnnSuccess = Process-Zip -ZipPath $CnnZip -ModelName "CNN" -RequiredWeightRegex "cnn_mobilenetv2\.(onnx|pt)"
}

# Cleanup temp dir
if (Test-Path $TempDir) { Remove-Item -Path $TempDir -Recurse -Force }

# 2. Print Installed Manifest & SHA-256 Hashes
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "📋 INSTALLED WEIGHTS & CHECKSUMS (model_weights/)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

$InstalledFiles = Get-ChildItem -Path $ModelWeightsDir -File | Where-Object { $_.Name -notmatch "^\." }
if ($InstalledFiles.Count -eq 0) {
    Write-Host "No files found in model_weights/." -ForegroundColor Red
} else {
    foreach ($file in $InstalledFiles) {
        $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash
        $sizeMB = [math]::Round($file.Length / 1MB, 2)
        Write-Host ("{0,-30} {1,8} MB | SHA256: {2}" -f $file.Name, $sizeMB, $hash) -ForegroundColor White
    }
}

# Trigger paper metrics auto-update if script exists
$paperScript = Join-Path $BaseDir "scripts\update_paper_metrics.py"
if (Test-Path $paperScript) {
    Write-Host "`n📝 Syncing installed metrics into IEEE research paper..." -ForegroundColor DarkGray
    & ".venv\Scripts\python.exe" $paperScript
}

Write-Host "`n✅ Artifact installation complete. Run 'python scripts\validate_models.py --strict' to verify." -ForegroundColor Green
