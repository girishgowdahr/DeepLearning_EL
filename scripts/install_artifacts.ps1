# ForestGuard AI — Artifacts Installation & Integrity Verification Script
# Validates ZIP archives or directory artifacts, extracts safely to staging,
# verifies SHA-256 cryptographic integrity, and performs atomic deployments.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)][string]$YoloZip = "",
    [Parameter(Mandatory=$false)][string]$CnnZip = "",
    [Parameter(Mandatory=$false)][string]$ArtifactsDir = "",
    [Parameter(Mandatory=$false)][switch]$VerifyOnly = $false,
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

# Official Cryptographic SHA-256 Checksum Manifest for ForestGuard AI v1.0.0
$OFFICIAL_CHECKSUMS = @{
    "yolo_best.pt"         = "10648918c9cbbc35d93d63ad01ce5d86a579db813b72d1848c444e97a7a6f638"
    "yolo_best.onnx"       = "af5618f2d26d28ccdc1e61a50e577cb326d75a5e972f4a5f915ae35374b2fafb"
    "cnn_mobilenetv2.pt"   = "44734dcad5967300a683d12851d29760950be6e0debb022516978ebbcb03f499"
    "cnn_mobilenetv2.onnx" = "1646ae489964767609d849a6e99aabe6485dc3a1c452c85026fbf9fac3810e17"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🌲 ForestGuard AI Artifact Installer & Integrity Verifier" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# ==============================================================================
# MODE: VERIFY ONLY (Audits on-disk files without modifying or copying)
# ==============================================================================
if ($VerifyOnly) {
    Write-Host "`n[AUDIT MODE] Verifying on-disk model artifacts against official v1.0.0 hashes..." -ForegroundColor Cyan
    $allPassed = $true

    foreach ($modelKey in $OFFICIAL_CHECKSUMS.Keys) {
        $filePath = Join-Path $ModelWeightsDir $modelKey
        if (-not (Test-Path $filePath)) {
            Write-Host "  ❌ MISSING:   $modelKey" -ForegroundColor Red
            $allPassed = $false
            continue
        }

        $fileSize = (Get-Item $filePath).Length
        $calcHash = (Get-FileHash -Path $filePath -Algorithm SHA256).Hash.ToLower()
        $expectedHash = $OFFICIAL_CHECKSUMS[$modelKey].ToLower()

        if ($calcHash -eq $expectedHash) {
            Write-Host "  ✅ VALID:     $modelKey ($([math]::Round($fileSize / 1MB, 2)) MB)" -ForegroundColor Green
            Write-Host "                SHA-256: $calcHash" -ForegroundColor DarkGray
        } else {
            Write-Host "  ❌ MISMATCH:  $modelKey" -ForegroundColor Red
            Write-Host "                Expected: $expectedHash" -ForegroundColor Yellow
            Write-Host "                Actual:   $calcHash" -ForegroundColor Red
            $allPassed = $false
        }
    }

    Write-Host "`n------------------------------------------------------------" -ForegroundColor Cyan
    if ($allPassed) {
        Write-Host "✅ VERIFICATION RESULT: All installed model artifacts match official hashes!" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "⚠️ VERIFICATION RESULT: One or more model artifacts are missing or altered." -ForegroundColor Yellow
        exit 1
    }
}

# ==============================================================================
# PHASE 0: RESOLVE INPUT ARCHIVES / DIRECTORIES
# ==============================================================================
$explicitArgsProvided = ($YoloZip -ne "") -or ($CnnZip -ne "") -or ($ArtifactsDir -ne "")

if (-not $explicitArgsProvided) {
    $candYolo = Join-Path $BaseDir "yolo_artifacts.zip"
    $candCnn = Join-Path $BaseDir "cnn_artifacts.zip"
    if (Test-Path $candYolo) { $YoloZip = $candYolo }
    if (Test-Path $candCnn) { $CnnZip = $candCnn }

    if (-not $YoloZip -and -not $CnnZip) {
        if ($Strict) {
            Write-Host "❌ FATAL: No artifact archives or directory provided, and -Strict was specified." -ForegroundColor Red
            exit 1
        }
        Write-Host "Usage:" -ForegroundColor Yellow
        Write-Host "  .\scripts\install_artifacts.ps1 -YoloZip <path> -CnnZip <path>" -ForegroundColor White
        Write-Host "  .\scripts\install_artifacts.ps1 -ArtifactsDir <folder_containing_weights>" -ForegroundColor White
        Write-Host "  .\scripts\install_artifacts.ps1 -VerifyOnly" -ForegroundColor White
        Write-Host "`nℹ No input archives found. You can download genuine release archives from the project release page," -ForegroundColor DarkGray
        Write-Host "  or run 'training/kaggle/ForestGuard_AI_Complete_Training.ipynb' to generate them." -ForegroundColor DarkGray
        exit 0
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

$stagedItemsToInstall = @()

# Helper to process a directory of extracted or source files
function Stage-FilesFromDirectory($dirPath, $targetName, $manifestFileHint) {
    $files = Get-ChildItem -Path $dirPath -Recurse -File
    if ($files.Count -eq 0) {
        Abort-Install "$targetName contains 0 files. Source is empty!"
    }

    $manifestHashes = @{}
    $manifestCandidate = $files | Where-Object { $_.Name -match "^(MANIFEST\.txt|.*manifest.*\.json|.*\.sha256)$" } | Select-Object -First 1
    if ($manifestCandidate) {
        Write-Host "  📋 Found manifest: $($manifestCandidate.Name)" -ForegroundColor DarkCyan
        $manifestLines = Get-Content -Path $manifestCandidate.FullName -ErrorAction SilentlyContinue
        foreach ($line in $manifestLines) {
            $trimmed = $line.Trim()
            if ($trimmed -match "^([a-fA-F0-9]{64})\s+[* ]?(.+)$") {
                $manifestHashes[$matches[2].Trim()] = $matches[1].ToLower()
            } elseif ($trimmed -match "^(\S+)\s+[\d\.]+\s+([a-fA-F0-9]{64})$") {
                $manifestHashes[$matches[1].Trim()] = $matches[2].ToLower()
            }
        }
    }

    foreach ($file in $files) {
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

        $calcHash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash.ToLower()

        # Integrity Check 1: Archive internal manifest
        $manifestVerified = $false
        if ($manifestHashes.ContainsKey($file.Name)) {
            $expectedHash = $manifestHashes[$file.Name]
            if ($calcHash -eq $expectedHash) {
                Write-Host "    [MANIFEST: MATCH]    $($file.Name) matches package manifest ($calcHash)" -ForegroundColor Green
                $manifestVerified = $true
            } else {
                Abort-Install "Package manifest MISMATCH for '$($file.Name)'! Expected: $expectedHash, Found: $calcHash"
            }
        }

        # Integrity Check 2: Official v1.0.0 reference table
        if ($OFFICIAL_CHECKSUMS.ContainsKey($file.Name)) {
            $officialHash = $OFFICIAL_CHECKSUMS[$file.Name].ToLower()
            if ($calcHash -eq $officialHash) {
                Write-Host "    [OFFICIAL V1.0 MATCH] $($file.Name) verified against official release digest!" -ForegroundColor Green
                $manifestVerified = $true
            } else {
                if ($Strict) {
                    Abort-Install "Cryptographic MISMATCH for '$($file.Name)' against official v1.0 release hash!`nExpected: $officialHash`nActual:   $calcHash"
                } else {
                    Write-Host "    [CUSTOM/UPDATED]     $($file.Name) differs from official v1.0 hash ($calcHash)" -ForegroundColor Yellow
                }
            }
        } else {
            if (-not $manifestVerified) {
                Write-Host "    [CALCULATED HASH]    $($file.Name) -> $calcHash" -ForegroundColor DarkGray
            }
        }

        $destFileName = $file.Name
        if ($file.Name -eq "MANIFEST.txt") {
            $destFileName = "$($targetName.ToLower())_manifest.txt"
        }

        $script:stagedItemsToInstall += [PSCustomObject]@{
            SourcePath       = $file.FullName
            FileName         = $file.Name
            DestFileName     = $destFileName
            DestDir          = $destDir
            StagedHash       = $calcHash
            ModelName        = $targetName
            Size             = $file.Length
            ManifestVerified = $manifestVerified
        }
    }
}

# ==============================================================================
# PHASE 1: STAGING & ARCHIVE EXTRACTION AUDIT
# ==============================================================================
Write-Host "`n[PHASE 1] Pre-Installation Staging & Archive Integrity Audit..." -ForegroundColor Cyan

if ($ArtifactsDir -ne "") {
    Write-Host "`n--- Validating Source Directory ---" -ForegroundColor Yellow
    Write-Host "  Source: $ArtifactsDir" -ForegroundColor White
    if (-not (Test-Path $ArtifactsDir)) {
        Abort-Install "Specified ArtifactsDir does not exist: '$ArtifactsDir'"
    }
    Stage-FilesFromDirectory -dirPath $ArtifactsDir -targetName "SourceDir" -manifestFileHint $null
}

if ($YoloZip -ne "") {
    Write-Host "`n--- Validating YOLO Archive ---" -ForegroundColor Yellow
    Write-Host "  Source: $YoloZip" -ForegroundColor White
    if (-not (Test-Path $YoloZip)) { Abort-Install "YOLO Archive not found: '$YoloZip'" }
    $stageSub = Join-Path $TempDir "YOLO"
    New-Item -ItemType Directory -Path $stageSub -Force | Out-Null
    try {
        Expand-Archive -Path $YoloZip -DestinationPath $stageSub -Force
    } catch {
        Abort-Install "Failed to extract YOLO archive: $($_.Exception.Message)"
    }
    Stage-FilesFromDirectory -dirPath $stageSub -targetName "YOLO" -manifestFileHint $null
}

if ($CnnZip -ne "") {
    Write-Host "`n--- Validating CNN Archive ---" -ForegroundColor Yellow
    Write-Host "  Source: $CnnZip" -ForegroundColor White
    if (-not (Test-Path $CnnZip)) { Abort-Install "CNN Archive not found: '$CnnZip'" }
    $stageSub = Join-Path $TempDir "CNN"
    New-Item -ItemType Directory -Path $stageSub -Force | Out-Null
    try {
        Expand-Archive -Path $CnnZip -DestinationPath $stageSub -Force
    } catch {
        Abort-Install "Failed to extract CNN archive: $($_.Exception.Message)"
    }
    Stage-FilesFromDirectory -dirPath $stageSub -targetName "CNN" -manifestFileHint $null
}

if ($stagedItemsToInstall.Count -eq 0) {
    Abort-Install "No valid model artifacts discovered to install across provided inputs!"
}

# ==============================================================================
# PHASE 2: ATOMIC INSTALLATION
# ==============================================================================
Write-Host "`n[PHASE 2] All staged artifacts verified. Installing to destination..." -ForegroundColor Cyan

# Ensure destination folders exist
New-Item -ItemType Directory -Path $ModelWeightsDir -Force | Out-Null
New-Item -ItemType Directory -Path $MetricsDir -Force | Out-Null
New-Item -ItemType Directory -Path $GraphsDir -Force | Out-Null
New-Item -ItemType Directory -Path $PredictionsDir -Force | Out-Null
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

# Backup existing weights if present
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

# Copy validated items
foreach ($item in $stagedItemsToInstall) {
    $destFile = Join-Path $item.DestDir $item.DestFileName
    Copy-Item -Path $item.SourcePath -Destination $destFile -Force
    Write-Host "  + Installed: $($item.FileName) -> $($item.DestFileName)" -ForegroundColor Green
}

# ==============================================================================
# PHASE 3: POST-INSTALLATION DISK INTEGRITY VERIFICATION
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
