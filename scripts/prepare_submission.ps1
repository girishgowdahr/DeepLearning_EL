# ForestGuard AI — Academic & Production Submission Packager
# Generates a pristine, clean submission bundle with research paper, documentation,
# architecture diagrams, actual test reports, source code, and demo instructions.
# Excludes virtual environments, caches, raw datasets, temporary files, and secrets.
param(
    [string]$OutputDir = "dist_submission",
    [switch]$CreateZip = $true
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$BaseDir = (Get-Item $PSScriptRoot).Parent.FullName
$TargetDir = Join-Path $BaseDir $OutputDir
$SubmissionRoot = Join-Path $TargetDir "ForestGuard_AI_Submission"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🌲 ForestGuard AI — Final Submission Package Generator" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Base Project:      $BaseDir"
Write-Host "Output Directory:  $SubmissionRoot`n"

# 1. Clean previous submission directory
if (Test-Path $TargetDir) {
    Write-Host "🧹 Purging previous submission directory..." -ForegroundColor DarkGray
    Remove-Item -Path $TargetDir -Recurse -Force
}
New-Item -ItemType Directory -Path $SubmissionRoot -Force | Out-Null

# Create destination subdirectories
$subPaper = Join-Path $SubmissionRoot "paper"
$subDocs  = Join-Path $SubmissionRoot "documentation"
$subReps  = Join-Path $SubmissionRoot "reports"
$subDemo  = Join-Path $SubmissionRoot "demo_instructions"
$subSrc   = Join-Path $SubmissionRoot "source_code"

foreach ($d in @($subPaper, $subDocs, $subReps, $subDemo, $subSrc)) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

# -----------------------------------------------------------------------------
# 2. Bundle Research Paper & LaTeX Assets
# -----------------------------------------------------------------------------
Write-Host "[1/5] Bundling IEEE Research Paper & Citations..." -ForegroundColor Yellow
$paperSrc = Join-Path $BaseDir "docs\paper"
if (Test-Path $paperSrc) {
    Copy-Item -Path "$paperSrc\*" -Destination $subPaper -Recurse -Force
    Write-Host "  ✓ Copied LaTeX source, bibliography, and markdown draft" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# 3. Bundle Documentation & Diagrams
# -----------------------------------------------------------------------------
Write-Host "[2/5] Bundling Architecture Docs & Diagrams..." -ForegroundColor Yellow
$docsSrc = Join-Path $BaseDir "docs"
$docFiles = Get-ChildItem -Path $docsSrc -File -ErrorAction SilentlyContinue
foreach ($f in $docFiles) {
    Copy-Item -Path $f.FullName -Destination $subDocs -Force
}
$diagramsSrc = Join-Path $docsSrc "diagrams"
if (Test-Path $diagramsSrc) {
    Copy-Item -Path $diagramsSrc -Destination $subDocs -Recurse -Force
}
Write-Host "  ✓ Copied architecture documentation and diagrams" -ForegroundColor Green

# -----------------------------------------------------------------------------
# 4. Bundle Actual Verification Reports & Metrics
# -----------------------------------------------------------------------------
Write-Host "[3/5] Bundling Actual Verification Reports & Audit Logs..." -ForegroundColor Yellow
$reportsToCopy = @(
    "results\final_verification_report.txt",
    "results\inference_test_report.txt",
    "results\metrics",
    "results\graphs",
    "results\predictions\inference_predictions.json"
)

foreach ($r in $reportsToCopy) {
    $srcPath = Join-Path $BaseDir $r
    if (Test-Path $srcPath) {
        $destPath = Join-Path $subReps (Split-Path $r -Leaf)
        Copy-Item -Path $srcPath -Destination $destPath -Recurse -Force
        Write-Host "  + Included Report/Metric: $r" -ForegroundColor Green
    }
}

# -----------------------------------------------------------------------------
# 5. Create Standalone Demo & Reproduction Guide
# -----------------------------------------------------------------------------
Write-Host "[4/5] Generating Demo Instructions & Reproduction Guide..." -ForegroundColor Yellow
$demoGuidePath = Join-Path $subDemo "DEMO_RUN_GUIDE.md"
$demoGuideLines = @(
    "# ForestGuard AI — Demonstration & Operational Guide",
    "====================================================",
    "Academic Prototype: Early Wildfire & Smoke Detection Decision System",
    "Authors: Deep Learning EL Project Team (Sem 3)",
    "",
    "## 1. Prerequisites",
    "- Windows 11 / 10 or Linux host",
    "- Python 3.11 with Virtual Environment (.venv)",
    "- Node.js >= 18.0 & npm",
    "- Cloudflare Tunnel binary (cloudflared) for Android Mobile Streaming",
    "",
    "## 2. Quick Local Start (VS Code or PowerShell)",
    "1. Open terminal in the project root:",
    '   cd "D:\RVCE\Sem3\Deep Learning EL\ForestGuard-AI"',
    "2. Launch the end-to-end demo launcher:",
    "   .\\scripts\\run_demo.ps1",
    "   This will:",
    "   - Verify Python and Node environments.",
    "   - Start the FastAPI backend on http://localhost:8000.",
    "   - Start or configure Cloudflare Quick Tunnel (TryCloudflare HTTPS).",
    "   - Launch the React dashboard at http://localhost:8000.",
    "",
    "## 3. Laptop Camera vs Android Phone Camera",
    "- Laptop Camera: Automatically used as default fallback when available.",
    "- Android Phone Camera:",
    "  1. Open http://localhost:8000/live in your browser.",
    "  2. Scan the generated TryCloudflare HTTPS QR Code with your Android phone.",
    "  3. Grant camera permission on mobile Chrome.",
    "  4. The phone streams live video over secure WebSockets (WSS) with sub-second latency.",
    "",
    "## 4. Model Verification & Health Check",
    "- Run the system health check:",
    "  .\\scripts\\health_check.ps1",
    "- Run the model inference & degradation test suite:",
    "  python scripts\\test_model_inference.py",
    "- Run the full verification suite:",
    "  .\\scripts\\finalize_forestguard.ps1"
)
Set-Content -Path $demoGuidePath -Value ($demoGuideLines -join "`r`n") -Encoding UTF8
Write-Host "  ✓ Created DEMO_RUN_GUIDE.md" -ForegroundColor Green

# -----------------------------------------------------------------------------
# 6. Copy Clean Source Code (Strictly excluding virtualenvs, caches, secrets)
# -----------------------------------------------------------------------------
Write-Host "[5/5] Packaging Clean Source Code..." -ForegroundColor Yellow

$sourceFolders = @("backend", "frontend", "scripts", "training")
foreach ($folder in $sourceFolders) {
    $srcF = Join-Path $BaseDir $folder
    $dstF = Join-Path $subSrc $folder
    if (-not (Test-Path $srcF)) { continue }

    New-Item -ItemType Directory -Path $dstF -Force | Out-Null
    
    # Copy files excluding caches, node_modules, .env, secrets
    Get-ChildItem -Path $srcF -Recurse | Where-Object {
        $_.FullName -notmatch "\\node_modules(\\|$)" -and
        $_.FullName -notmatch "\\__pycache__(\\|$)" -and
        $_.FullName -notmatch "\\\.pytest_cache(\\|$)" -and
        $_.FullName -notmatch "\\\.git(\\|$)" -and
        $_.FullName -notmatch "\\\.env(\.local|\.production|\.development)?$" -and
        $_.FullName -notmatch "\.(pyc|pyo|pem|key|log|db)$"
    } | ForEach-Object {
        $relPath = $_.FullName.Substring($srcF.Length).TrimStart("\/")
        $targetPath = Join-Path $dstF $relPath
        if ($_.PSIsContainer) {
            New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
        } else {
            $parentDir = Split-Path $targetPath -Parent
            if (-not (Test-Path $parentDir)) { New-Item -ItemType Directory -Path $parentDir -Force | Out-Null }
            Copy-Item -Path $_.FullName -Destination $targetPath -Force
        }
    }
    Write-Host "  ✓ Packaged clean $folder/" -ForegroundColor Green
}

# Copy root config files
$rootFiles = @("README.md", "package.json", ".gitignore", "LICENSE")
foreach ($rf in $rootFiles) {
    $p = Join-Path $BaseDir $rf
    if (Test-Path $p) {
        Copy-Item -Path $p -Destination $SubmissionRoot -Force
    }
}

# -----------------------------------------------------------------------------
# 7. Automated Security & Secret Sanity Scan
# -----------------------------------------------------------------------------
Write-Host "`n🔒 Executing Automated Security Audit on Submission Files..." -ForegroundColor Cyan
$leakedSecrets = Get-ChildItem -Path $SubmissionRoot -Recurse -File | Where-Object {
    $_.Name -match "^(\.env(\.local|\.production|\.development)?)$" -or
    $_.Name -match "\.(pem|key)$" -or
    $_.Name -match "token.*\.txt$" -or
    $_.FullName -match "node_modules|\.venv|__pycache__"
}

if ($leakedSecrets.Count -gt 0) {
    Write-Host "❌ CRITICAL SECURITY ERROR: Forbidden or sensitive files found in submission package!" -ForegroundColor Red
    foreach ($ls in $leakedSecrets) {
        Write-Host "   - $($ls.FullName)" -ForegroundColor Red
    }
    Remove-Item -Path $TargetDir -Recurse -Force
    exit 1
}
Write-Host "  ✅ Security Audit Passed: Zero private keys, tokens, virtualenvs, or caches present." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 8. Generate Cryptographic Manifest
# -----------------------------------------------------------------------------
Write-Host "`n📋 Computing Cryptographic SHA-256 Manifest..." -ForegroundColor Cyan
$manifestLines = @(
    "# ForestGuard AI — Academic Submission Manifest",
    "# Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss (UTCzzz)')",
    "# Algorithm: SHA-256",
    "# -----------------------------------------------------------------------------"
)

$allSubmissionFiles = Get-ChildItem -Path $SubmissionRoot -Recurse -File
foreach ($file in $allSubmissionFiles) {
    $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash.ToLower()
    $rel = $file.FullName.Substring($SubmissionRoot.Length).TrimStart("\/").Replace("\", "/")
    $manifestLines += "$hash *$rel"
}

$manifestPath = Join-Path $SubmissionRoot "SUBMISSION_MANIFEST.sha256"
Set-Content -Path $manifestPath -Value ($manifestLines -join "`n") -Encoding UTF8
Write-Host "  ✓ Recorded $(($allSubmissionFiles.Count + 1)) verified files to SUBMISSION_MANIFEST.sha256" -ForegroundColor Green

# -----------------------------------------------------------------------------
# 9. Create Zip Archive if requested
# -----------------------------------------------------------------------------
if ($CreateZip) {
    $zipDest = Join-Path $TargetDir "ForestGuard_AI_Academic_Submission.zip"
    Write-Host "`n🗜️ Creating Standalone Submission Archive..." -ForegroundColor Cyan
    if (Test-Path $zipDest) { Remove-Item $zipDest -Force }
    Compress-Archive -Path "$SubmissionRoot\*" -DestinationPath $zipDest -Force
    $zipSize = [math]::Round((Get-Item $zipDest).Length / 1MB, 2)
    $zipHash = (Get-FileHash -Path $zipDest -Algorithm SHA256).Hash.ToLower()
    Write-Host "  ✅ Successfully created: $zipDest ($zipSize MB)" -ForegroundColor Green
    Write-Host "     SHA-256: $zipHash" -ForegroundColor DarkCyan
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "🎉 SUBMISSION PACKAGE GENERATED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Submission Directory: $SubmissionRoot"
Write-Host "Standalone Archive:   $(Join-Path $TargetDir 'ForestGuard_AI_Academic_Submission.zip')`n"
exit 0
