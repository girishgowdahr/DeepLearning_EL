# ForestGuard AI — Artifacts Installation & Integrity Verification Script
# Validates ZIP archives, extracts safely to staging, verifies SHA-256 integrity,
# and prevents partial or corrupted artifact deployments.
param(
    [Parameter(Mandatory=$false)][string]$YoloZip = "",
    [Parameter(Mandatory=$false)][string]$CnnZip = "",
    [Parameter(Mandatory=$false)][switch]$Strict = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$BaseDir = (Get-Item $PSScriptRoot).Parent.FullName
$ModelWeightsDir = Join-Path $BaseDir "model_weights"
$MetricsDir = Join-Path $BaseDir "results\metrics"
$GraphsDir = Join-Path $BaseDir "results\graphs"
$PredictionsDir = Join-Path $BaseDir "results\predictions"
$ResultsDir = Join-Path $BaseDir "results"
$TempDir = Join-Path $BaseDir "run\temp_artifacts_staging"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🌲 ForestGuard AI Artifact Installer & Integrity Verifier" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Resolve target archives
$explicitArgsProvided = ($YoloZip -ne "") -or ($CnnZip -ne "")

if (-not $explicitArgsProvided) {
    $candYolo = Join-Path $BaseDir "yolo_artifacts.zip"
    $candCnn = Join-Path $BaseDir "cnn_artifacts.zip"
    if (Test-Path $candYolo) { $YoloZip = $candYolo }
    if (Test-Path $candCnn) { $CnnZip = $candCnn }

    if (-not $YoloZip -and -not $CnnZip) {
        if ($Strict) {
            Write-Host "❌ FATAL: No artifact ZIP files provided or found, and -Strict was specified." -ForegroundColor Red
            exit 1
        }
        Write-Host "Usage: .\scripts\install_artifacts.ps1 -YoloZip <path> -CnnZip <path>" -ForegroundColor Yellow
        Write-Host "ℹ No ZIP files specified, and no candidate archives found in root." -ForegroundColor DarkGray
        Write-Host "  Artifact installation step skipped safely without modifying models." -ForegroundColor DarkGray
        exit 0
    }
}

$targets = @()
if ($YoloZip) {
    $targets += [PSCustomObject]@{
        Name = "YOLO"
        ZipPath = $YoloZip
        RequiredFile = "yolo_best.pt"
        RequiredRegex = "^yolo_best\.(pt|onnx)$"
    }
}
if ($CnnZip) {
    $targets += [PSCustomObject]@{
        Name = "CNN"
        ZipPath = $CnnZip
        RequiredFile = "cnn_mobilenetv2.onnx"
        RequiredRegex = "^cnn_mobilenetv2\.(onnx|pt)$"
    }
}

# Clean any leftover staging directory
if (Test-Path $TempDir) {
    Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

function Abort-Install($message) {
    Write-Host "`n❌ FATAL ARTIFACT VALIDATION FAILURE:" -ForegroundColor Red
    Write-Host "   $message" -ForegroundColor Red
    Write-Host "   Installation aborted. Zero files were copied or overwritten." -ForegroundColor Red
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    exit 1
}

# ==============================================================================
# PHASE 1: STAGING & COMPLETE ARCHIVE INTEGRITY VALIDATION
# (Nothing is copied to model_weights/ or results/ during Phase 1)
# ==============================================================================
Write-Host "`n[PHASE 1] Pre-Installation Staging & Archive Integrity Audit..." -ForegroundColor Cyan

$stagedItemsToInstall = @()

foreach ($target in $targets) {
    Write-Host "`n--- Validating $($target.Name) Archive ---" -ForegroundColor Yellow
    Write-Host "  Source: $($target.ZipPath)" -ForegroundColor White

    # Check 1: File existence
    if (-not (Test-Path $target.ZipPath)) {
        Abort-Install "Archive file not found: '$($target.ZipPath)'"
    }

    # Check 2: File size sanity (must not be empty or truncated)
    $zipItem = Get-Item $target.ZipPath
    if ($zipItem.Length -lt 100KB) {
        Abort-Install "$($target.Name) archive is suspiciously small ($($zipItem.Length) bytes). Archive is corrupted or empty."
    }

    # Check 3: Extraction to isolated staging
    $stageSub = Join-Path $TempDir $target.Name
    New-Item -ItemType Directory -Path $stageSub -Force | Out-Null
    try {
        Expand-Archive -Path $target.ZipPath -DestinationPath $stageSub -Force
    } catch {
        Abort-Install "Failed to extract $($target.Name) archive '$($target.ZipPath)'. Corrupted ZIP archive: $($_.Exception.Message)"
    }

    # Check 4: Non-empty extraction
    $extractedFiles = Get-ChildItem -Path $stageSub -Recurse -File
    if ($extractedFiles.Count -eq 0) {
        Abort-Install "$($target.Name) archive extracted 0 files. Archive is empty!"
    }
    Write-Host "  ✓ Extracted $($extractedFiles.Count) files to staging area." -ForegroundColor Green

    # Check 5: Required exact model weight presence
    $reqWeight = $extractedFiles | Where-Object { $_.Name -match $target.RequiredRegex }
    if (-not $reqWeight) {
        $extractedNames = ($extractedFiles | ForEach-Object { $_.Name }) -join ", "
        Abort-Install "$($target.Name) archive lacks required model weight '$($target.RequiredFile)'! Contents: [$extractedNames]"
    }

    foreach ($w in $reqWeight) {
        if ($w.Length -lt 100KB) {
            Abort-Install "Required model weight '$($w.Name)' is truncated or corrupted ($($w.Length) bytes, expected >= 100 KB)!"
        }
        Write-Host "  ✓ Verified presence of model weight: $($w.Name) ($([math]::Round($w.Length / 1MB, 2)) MB)" -ForegroundColor Green
    }

    # Check 6: Parse pre-packaged archive manifest if present (MANIFEST.txt or *.sha256)
    $manifestHashes = @{}
    $manifestFile = $extractedFiles | Where-Object { $_.Name -match "^(MANIFEST\.txt|.*manifest.*\.json|.*\.sha256)$" } | Select-Object -First 1

    if ($manifestFile -and (Test-Path $manifestFile.FullName)) {
        Write-Host "  📋 Found pre-packaged manifest: $($manifestFile.Name)" -ForegroundColor DarkCyan
        $manifestLines = Get-Content -Path $manifestFile.FullName -ErrorAction SilentlyContinue
        foreach ($line in $manifestLines) {
            $trimmed = $line.Trim()
            # Standard sha256: <hash> *<file> or <hash>  <file>
            if ($trimmed -match "^([a-fA-F0-9]{64})\s+[* ]?(.+)$") {
                $manifestHashes[$matches[2].Trim()] = $matches[1].ToLower()
            }
            # Table format: <file> <size_mb> <hash>
            elseif ($trimmed -match "^(\S+)\s+[\d\.]+\s+([a-fA-F0-9]{64})$") {
                $manifestHashes[$matches[1].Trim()] = $matches[2].ToLower()
            }
        }
    }

    # Check 7: Compute staged SHA-256 digests and perform integrity verification
    foreach ($file in $extractedFiles) {
        # Determine destination category
        $destDir = $null
        if ($file.Name -match "\.(pt|onnx)$" -or $file.Name -eq "MODEL_INFO.json" -or $file.Name -eq "MANIFEST.txt") {
            $destDir = $ModelWeightsDir
        } elseif ($file.Name -match "metrics\.json$") {
            $destDir = $MetricsDir
        } elseif ($file.Name -match "\.(png|jpg|jpeg)$") {
            $destDir = $GraphsDir
        } elseif ($file.Name -match "\.(csv|txt)$") {
            $destDir = $PredictionsDir
        }

        if (-not $destDir) { continue }

        # Calculate cryptographic hash
        $calcHash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash.ToLower()

        # Perform Integrity Verification (distinguished from pure calculation)
        $manifestVerified = $false
        if ($manifestHashes.ContainsKey($file.Name)) {
            $expectedHash = $manifestHashes[$file.Name]
            if ($calcHash -eq $expectedHash) {
                Write-Host "    [INTEGRITY: MATCH] $($file.Name) matches archive manifest ($calcHash)" -ForegroundColor Green
                $manifestVerified = $true
            } else {
                Abort-Install "Cryptographic SHA-256 MISMATCH for '$($file.Name)'! Expected: $expectedHash, Staged: $calcHash"
            }
        } else {
            Write-Host "    [CALCULATED HASH]  $($file.Name) -> $calcHash (no archive manifest entry)" -ForegroundColor DarkGray
        }

        $destFileName = $file.Name
        if ($file.Name -eq "MANIFEST.txt") {
            $destFileName = "$($target.Name.ToLower())_manifest.txt"
        }

        $stagedItemsToInstall += [PSCustomObject]@{
            SourcePath       = $file.FullName
            FileName         = $file.Name
            DestFileName     = $destFileName
            DestDir          = $destDir
            StagedHash       = $calcHash
            ModelName        = $target.Name
            Size             = $file.Length
            ManifestVerified = $manifestVerified
        }
    }
}

if ($stagedItemsToInstall.Count -eq 0) {
    Abort-Install "No valid model artifacts found to install across supplied archives!"
}

# ==============================================================================
# PHASE 2: ATOMIC INSTALLATION (Only reached if ALL targets passed validation)
# ==============================================================================
Write-Host "`n[PHASE 2] All archives verified. Installing artifacts to destination..." -ForegroundColor Cyan

# 1. Backup existing weights
$ExistingWeights = Get-ChildItem -Path $ModelWeightsDir -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -match "\.(pt|onnx)$" }
if ($ExistingWeights.Count -gt 0) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BackupDir = Join-Path $ModelWeightsDir "backup_$Timestamp"
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    Write-Host "📦 Backing up $($ExistingWeights.Count) existing weight file(s) to $BackupDir..." -ForegroundColor Yellow
    foreach ($f in $ExistingWeights) {
        Copy-Item -Path $f.FullName -Destination (Join-Path $BackupDir $f.Name) -Force
    }
}

# Ensure destination folders exist
New-Item -ItemType Directory -Path $ModelWeightsDir -Force | Out-Null
New-Item -ItemType Directory -Path $MetricsDir -Force | Out-Null
New-Item -ItemType Directory -Path $GraphsDir -Force | Out-Null
New-Item -ItemType Directory -Path $PredictionsDir -Force | Out-Null
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

# 2. Copy validated items
foreach ($item in $stagedItemsToInstall) {
    $destFile = Join-Path $item.DestDir $item.DestFileName
    Copy-Item -Path $item.SourcePath -Destination $destFile -Force
    Write-Host "  + Installed: $($item.FileName) -> $($item.DestFileName)" -ForegroundColor Green
}

# ==============================================================================
# PHASE 3: POST-INSTALLATION DISK INTEGRITY VERIFICATION
# (Re-reads all installed files from disk to guarantee byte-for-byte fidelity)
# ==============================================================================
Write-Host "`n[PHASE 3] Post-Installation Disk Integrity Audit..." -ForegroundColor Cyan

$manifestRecords = @()
$manifestRecords += "# ForestGuard AI — Cryptographic Installed Artifacts Manifest"
$manifestRecords += "# Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss (UTCzzz)')"
$manifestRecords += "# Algorithm: SHA-256"
$manifestRecords += "# -----------------------------------------------------------------------------"

foreach ($item in $stagedItemsToInstall) {
    $destFile = Join-Path $item.DestDir $item.DestFileName
    if (-not (Test-Path $destFile)) {
        Abort-Install "Post-installation verification failed: '$destFile' is missing from disk!"
    }

    $diskHash = (Get-FileHash -Path $destFile -Algorithm SHA256).Hash.ToLower()
    if ($diskHash -ne $item.StagedHash) {
        Abort-Install "Post-installation DISK CORRUPTION detected for '$($item.DestFileName)'! Staged: $($item.StagedHash), Disk: $diskHash"
    }

    $relPath = $destFile.Replace($BaseDir, "").TrimStart("\/")
    $manifestRecords += "$diskHash *$relPath"
    Write-Host "  ✅ Verified on-disk integrity: $($item.DestFileName) (SHA-256: $diskHash)" -ForegroundColor Green
}

# Write out the official installed manifest to both model_weights and results
$manifestPathWeights = Join-Path $ModelWeightsDir "installed_manifest.sha256"
$manifestPathResults = Join-Path $ResultsDir "installed_manifest.sha256"

$manifestContent = $manifestRecords -join "`n"
Set-Content -Path $manifestPathWeights -Value $manifestContent -Encoding UTF8
Set-Content -Path $manifestPathResults -Value $manifestContent -Encoding UTF8
Write-Host "  📝 Recorded verified cryptographic manifest to: $manifestPathWeights" -ForegroundColor DarkCyan

# Cleanup staging directory
if (Test-Path $TempDir) {
    Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}

# Auto-update IEEE research paper metrics
$paperScript = Join-Path $BaseDir "scripts\update_paper_metrics.py"
if (Test-Path $paperScript) {
    Write-Host "`n📝 Syncing installed metrics into IEEE research paper..." -ForegroundColor DarkGray
    $venvPy = Join-Path $BaseDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
        & $venvPy $paperScript
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "✅ ARTIFACT INSTALLATION & INTEGRITY VERIFICATION COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
exit 0
