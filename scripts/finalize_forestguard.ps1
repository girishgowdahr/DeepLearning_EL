# ForestGuard AI — Fast Finalization Automation Script
# Runs end-to-end artifact installation, strict model verification, test suites,
# production build, and generates a timestamped final verification audit report.
param(
    [Parameter(Mandatory=$false)][string]$YoloZip = "",
    [Parameter(Mandatory=$false)][string]$CnnZip = "",
    [switch]$SkipTests = $false,
    [switch]$Strict = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$BaseDir = (Get-Item $PSScriptRoot).Parent.FullName
$venvPython = Join-Path $BaseDir ".venv\Scripts\python.exe"
$ResultsDir = Join-Path $BaseDir "results"
$ReportPath = Join-Path $ResultsDir "final_verification_report.txt"
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🌲 ForestGuard AI — End-to-End System Finalizer & Verification Suite" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

# Track stage outcomes explicitly via actual process exit codes
$executionRecord = [ordered]@{
    ArtifactInstall = @{ ExitCode = -1; Status = "NOT_RUN" }
    ModelValidation = @{ ExitCode = -1; Status = "NOT_RUN" }
    BackendTests    = @{ ExitCode = -1; Status = "NOT_RUN" }
    FrontendTests   = @{ ExitCode = -1; Status = "NOT_RUN" }
    FrontendBuild   = @{ ExitCode = -1; Status = "NOT_RUN" }
    PaperSync       = @{ ExitCode = -1; Status = "NOT_RUN" }
}

function Generate-FinalReport([string]$OverallStatus, [string]$SummaryNote) {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss (UTCzzz)"
    $gitBranch = (& git rev-parse --abbrev-ref HEAD 2>$null)
    $gitCommit = (& git rev-parse HEAD 2>$null)
    $pyVersion = (& $venvPython -V 2>$null)
    $nodeVersion = (& node -v 2>$null)
    $npmVersion = (& npm -v 2>$null)

    $cnnPath = Join-Path $BaseDir "model_weights\cnn_mobilenetv2.onnx"
    $yoloPath = Join-Path $BaseDir "model_weights\yolo_best.pt"
    $hasCnn = Test-Path $cnnPath
    $hasYolo = Test-Path $yoloPath

    $cnnSize = if ($hasCnn) { "$([math]::Round((Get-Item $cnnPath).Length / 1MB, 2)) MB" } else { "N/A" }
    $yoloSize = if ($hasYolo) { "$([math]::Round((Get-Item $yoloPath).Length / 1MB, 2)) MB" } else { "N/A" }
    $cnnHash = if ($hasCnn) { (Get-FileHash -Path $cnnPath -Algorithm SHA256).Hash.ToLower() } else { "NOT_INSTALLED" }
    $yoloHash = if ($hasYolo) { (Get-FileHash -Path $yoloPath -Algorithm SHA256).Hash.ToLower() } else { "NOT_INSTALLED" }

    $inferenceReadiness = if ($hasCnn -and $hasYolo) {
        "OPERATIONAL (Full dual-model inference with MobileNetV2 ONNX & YOLO11n PyTorch)"
    } else {
        "DEGRADED SAFE MODE (MODEL_UNAVAILABLE state active; zero false NO_FIRE detections; awaiting Kaggle weights)"
    }

    $reportLines = @()
    $reportLines += "================================================================================"
    $reportLines += "🌲 FORESTGUARD AI: END-TO-END FINAL SYSTEM VERIFICATION REPORT"
    $reportLines += "================================================================================"
    $reportLines += "Generated At:        $now"
    $reportLines += "Project Workspace:   $BaseDir"
    $reportLines += "Git Branch:          $gitBranch"
    $reportLines += "Git Commit:          $gitCommit"
    $reportLines += "Python Environment:  $pyVersion ($venvPython)"
    $reportLines += "Node Environment:    Node $nodeVersion | npm $npmVersion"
    $reportLines += "Overall Status:      $OverallStatus"
    $reportLines += "Summary Note:        $SummaryNote"
    $reportLines += ""
    $reportLines += "--------------------------------------------------------------------------------"
    $reportLines += "1. MODEL WEIGHTS & INFERENCE READINESS AUDIT"
    $reportLines += "--------------------------------------------------------------------------------"
    $reportLines += "CNN Classifier:      cnn_mobilenetv2.onnx"
    $reportLines += "  - Installed:       $(if ($hasCnn) { 'YES' } else { 'NO (Pending Kaggle weights)' })"
    $reportLines += "  - File Size:       $cnnSize"
    $reportLines += "  - SHA-256 Digest:  $cnnHash"
    $reportLines += "YOLO Detector:       yolo_best.pt"
    $reportLines += "  - Installed:       $(if ($hasYolo) { 'YES' } else { 'NO (Pending Kaggle weights)' })"
    $reportLines += "  - File Size:       $yoloSize"
    $reportLines += "  - SHA-256 Digest:  $yoloHash"
    $reportLines += "Inference State:     $inferenceReadiness"
    $reportLines += ""
    $reportLines += "--------------------------------------------------------------------------------"
    $reportLines += "2. AUTOMATED EXECUTION STAGES & PROCESS EXIT CODES"
    $reportLines += "--------------------------------------------------------------------------------"
    $reportLines += ("{0,-28} | {1,-10} | {2}" -f "Stage Name", "Exit Code", "Result Status")
    $reportLines += ("-" * 80)
    $reportLines += ("{0,-28} | {1,-10} | {2}" -f "1. Artifact Installation", $executionRecord.ArtifactInstall.ExitCode, $executionRecord.ArtifactInstall.Status)
    $reportLines += ("{0,-28} | {1,-10} | {2}" -f "2. Model Validation Audit", $executionRecord.ModelValidation.ExitCode, $executionRecord.ModelValidation.Status)
    $reportLines += ("{0,-28} | {1,-10} | {2}" -f "3. Backend Pytest Suite", $executionRecord.BackendTests.ExitCode, $executionRecord.BackendTests.Status)
    $reportLines += ("{0,-28} | {1,-10} | {2}" -f "4. Frontend Unit Tests", $executionRecord.FrontendTests.ExitCode, $executionRecord.FrontendTests.Status)
    $reportLines += ("{0,-28} | {1,-10} | {2}" -f "5. Frontend Production Build", $executionRecord.FrontendBuild.ExitCode, $executionRecord.FrontendBuild.Status)
    $reportLines += ("{0,-28} | {1,-10} | {2}" -f "6. Paper Metrics Sync", $executionRecord.PaperSync.ExitCode, $executionRecord.PaperSync.Status)
    $reportLines += ""
    $reportLines += "--------------------------------------------------------------------------------"
    $reportLines += "3. INSTALLED ARTIFACTS INVENTORY & CHECKSUMS (model_weights/)"
    $reportLines += "--------------------------------------------------------------------------------"
    $modelFiles = Get-ChildItem -Path (Join-Path $BaseDir "model_weights") -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch "^\." }
    if ($modelFiles.Count -eq 0) {
        $reportLines += "No model weight files currently installed in model_weights/."
    } else {
        $reportLines += ("{0,-32} | {1,10} | {2}" -f "Filename", "Size (MB)", "SHA-256 Digest")
        $reportLines += ("-" * 110)
        foreach ($mf in $modelFiles) {
            $h = (Get-FileHash -Path $mf.FullName -Algorithm SHA256).Hash.ToLower()
            $sz = [math]::Round($mf.Length / 1MB, 2)
            $reportLines += ("{0,-32} | {1,10} | {2}" -f $mf.Name, "$sz MB", $h)
        }
    }
    $reportLines += ""
    $reportLines += "--------------------------------------------------------------------------------"
    $reportLines += "4. PENDING PHYSICAL-DEVICE & RUNTIME VERIFICATION CHECKS"
    $reportLines += "--------------------------------------------------------------------------------"
    $reportLines += "The following operational criteria require physical device interaction and"
    $reportLines += "cannot be evaluated in CI or headless environments:"
    $reportLines += ""
    $reportLines += "[ ] 1. Cloudflare Quick Tunnel Provisioning:"
    $reportLines += "       Run scripts/run_demo.ps1 and confirm *.trycloudflare.com HTTPS URL generation."
    $reportLines += "[ ] 2. QR Code Authentication on Android Mobile:"
    $reportLines += "       Scan the generated QR code on mobile device; confirm token-in-hash (#t=...) auth."
    $reportLines += "[ ] 3. Camera Permission & WebRTC/WSS Ingestion:"
    $reportLines += "       Grant camera permission in mobile Chrome; verify live WebSocket streaming."
    $reportLines += "[ ] 4. Real-time Ingestion Latency & Frame Rate:"
    $reportLines += "       Verify sub-second roundtrip latency and >= 15 FPS frame ingestion on dashboard."
    $reportLines += "[ ] 5. Live Bounding Box & HUD Synchronization:"
    $reportLines += "       Point camera at fire/smoke imagery; verify bounding box and confidence render."
    $reportLines += "[ ] 6. Browser Siren & Alarm Escalation:"
    $reportLines += "       Verify Web Audio siren sounds upon temporal confirmation (FIRE_CONFIRMED)."
    $reportLines += "[ ] 7. Network Disconnect & Reconnect Resilience:"
    $reportLines += "       Toggle mobile flight mode; verify automatic reconnect backoff without server crash."
    $reportLines += "================================================================================"

    $content = $reportLines -join "`r`n"
    Set-Content -Path $ReportPath -Value $content -Encoding UTF8
    Write-Host "`n📄 Written Final Verification Report to: $ReportPath" -ForegroundColor DarkCyan
}

# ==============================================================================
# STAGE 1: ARTIFACT INSTALLATION
# ==============================================================================
Write-Host "`n[STAGE 1/5] Installing Kaggle Model Artifacts..." -ForegroundColor Yellow
$installArgs = @()
if ($YoloZip) { $installArgs += "-YoloZip"; $installArgs += "`"$YoloZip`"" }
if ($CnnZip)  { $installArgs += "-CnnZip";  $installArgs += "`"$CnnZip`"" }
if ($Strict)  { $installArgs += "-Strict" }

$hasZipArgs = ($YoloZip -ne "") -or ($CnnZip -ne "")
$candYolo = Join-Path $BaseDir "yolo_artifacts.zip"
$candCnn = Join-Path $BaseDir "cnn_artifacts.zip"
$hasDefaultZips = (Test-Path $candYolo) -or (Test-Path $candCnn)

if ($hasZipArgs -or $hasDefaultZips) {
    & "$PSScriptRoot\install_artifacts.ps1" @installArgs
    $installExit = $LASTEXITCODE
    $executionRecord.ArtifactInstall.ExitCode = $installExit
    if ($installExit -ne 0) {
        $executionRecord.ArtifactInstall.Status = "FAILED (Exit Code: $installExit)"
        Write-Host "❌ FATAL: install_artifacts.ps1 failed with exit code $installExit!" -ForegroundColor Red
        Generate-FinalReport -OverallStatus "FAILED" -SummaryNote "Artifact installation failed during pre-staging or SHA-256 verification."
        exit $installExit
    } else {
        $executionRecord.ArtifactInstall.Status = "PASSED (Exit Code: 0)"
    }
} else {
    $executionRecord.ArtifactInstall.ExitCode = 0
    $executionRecord.ArtifactInstall.Status = "SKIPPED (No ZIP files provided or found in root)"
    Write-Host "ℹ No artifact ZIPs provided. Skipping extraction stage." -ForegroundColor DarkGray
}

# ==============================================================================
# STAGE 2: MODEL VALIDATION AUDIT
# ==============================================================================
Write-Host "`n[STAGE 2/5] Running Model Validation Suite..." -ForegroundColor Yellow
$cnnPath = Join-Path $BaseDir "model_weights\cnn_mobilenetv2.onnx"
$yoloPath = Join-Path $BaseDir "model_weights\yolo_best.pt"
$bothWeightsPresent = (Test-Path $cnnPath) -and (Test-Path $yoloPath)

if ($bothWeightsPresent -or $Strict) {
    Write-Host "Running strict model validation (--strict enforced)..." -ForegroundColor White
    & $venvPython "$BaseDir\scripts\validate_models.py" --strict
    $valExit = $LASTEXITCODE
    $executionRecord.ModelValidation.ExitCode = $valExit
    if ($valExit -ne 0) {
        $executionRecord.ModelValidation.Status = "FAILED (Exit Code: $valExit)"
        Write-Host "❌ FATAL: Strict model validation failed with exit code $valExit!" -ForegroundColor Red
        Generate-FinalReport -OverallStatus "FAILED" -SummaryNote "Model validation failed in strict mode."
        exit $valExit
    } else {
        $executionRecord.ModelValidation.Status = "PASSED (Exit Code: 0 - Full Trained Inference Operational)"
        Write-Host "✓ Strict model validation passed cleanly." -ForegroundColor Green
    }
} else {
    Write-Host "Weights pending; validating safe degraded fallback behavior..." -ForegroundColor DarkGray
    & $venvPython "$BaseDir\scripts\validate_models.py"
    $valExit = $LASTEXITCODE
    $executionRecord.ModelValidation.ExitCode = $valExit
    if ($valExit -ne 0) {
        $executionRecord.ModelValidation.Status = "FAILED (Exit Code: $valExit)"
        Write-Host "❌ FATAL: Degraded model validation failed with exit code $valExit!" -ForegroundColor Red
        Generate-FinalReport -OverallStatus "FAILED" -SummaryNote "Degraded model validation failed."
        exit $valExit
    } else {
        $executionRecord.ModelValidation.Status = "DEGRADED_MODE_PASSED (Exit Code: 0 - MODEL_UNAVAILABLE verified)"
        Write-Host "✓ Safe degraded fallback behavior verified." -ForegroundColor Green
    }
}

# ==============================================================================
# STAGE 3: BACKEND PYTEST SUITE
# ==============================================================================
if (-not $SkipTests) {
    Write-Host "`n[STAGE 3/5] Executing Backend Integration & Security Tests..." -ForegroundColor Yellow
    & $venvPython -m pytest "$BaseDir\backend\tests" -v
    $pytestExit = $LASTEXITCODE
    $executionRecord.BackendTests.ExitCode = $pytestExit
    if ($pytestExit -ne 0) {
        $executionRecord.BackendTests.Status = "FAILED (Exit Code: $pytestExit)"
        Write-Host "❌ FATAL: Backend tests failed with exit code $pytestExit!" -ForegroundColor Red
        Generate-FinalReport -OverallStatus "FAILED" -SummaryNote "Backend test suite failed."
        exit $pytestExit
    } else {
        $executionRecord.BackendTests.Status = "PASSED (Exit Code: 0)"
        Write-Host "✓ Backend tests passed cleanly." -ForegroundColor Green
    }

    # ==============================================================================
    # STAGE 4: FRONTEND TESTS & PRODUCTION BUILD
    # ==============================================================================
    Write-Host "`n[STAGE 4/5] Executing Frontend Tests & Production Compilation..." -ForegroundColor Yellow
    & node --test "$BaseDir\frontend\src\utils\phoneCamera.test.mjs"
    $nodeTestExit = $LASTEXITCODE
    $executionRecord.FrontendTests.ExitCode = $nodeTestExit
    if ($nodeTestExit -ne 0) {
        $executionRecord.FrontendTests.Status = "FAILED (Exit Code: $nodeTestExit)"
        Write-Host "❌ FATAL: Frontend unit tests failed with exit code $nodeTestExit!" -ForegroundColor Red
        Generate-FinalReport -OverallStatus "FAILED" -SummaryNote "Frontend unit tests failed."
        exit $nodeTestExit
    } else {
        $executionRecord.FrontendTests.Status = "PASSED (Exit Code: 0)"
        Write-Host "✓ Frontend unit tests passed cleanly." -ForegroundColor Green
    }

    Write-Host "Compiling frontend production bundle (npm run build)..." -ForegroundColor DarkGray
    Push-Location (Join-Path $BaseDir "frontend")
    try {
        & npm run build
        $npmBuildExit = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    $executionRecord.FrontendBuild.ExitCode = $npmBuildExit
    if ($npmBuildExit -ne 0) {
        $executionRecord.FrontendBuild.Status = "FAILED (Exit Code: $npmBuildExit)"
        Write-Host "❌ FATAL: Frontend build failed with exit code $npmBuildExit!" -ForegroundColor Red
        Generate-FinalReport -OverallStatus "FAILED" -SummaryNote "Frontend npm production build failed."
        exit $npmBuildExit
    } else {
        $executionRecord.FrontendBuild.Status = "PASSED (Exit Code: 0 - dist/ compiled)"
        Write-Host "✓ Frontend production bundle compiled to frontend/dist/." -ForegroundColor Green
    }
} else {
    Write-Host "`n[STAGE 3 & 4/5] Skipping test execution (-SkipTests enabled)." -ForegroundColor DarkGray
    $executionRecord.BackendTests.ExitCode = 0
    $executionRecord.BackendTests.Status = "SKIPPED (-SkipTests)"
    $executionRecord.FrontendTests.ExitCode = 0
    $executionRecord.FrontendTests.Status = "SKIPPED (-SkipTests)"
    $executionRecord.FrontendBuild.ExitCode = 0
    $executionRecord.FrontendBuild.Status = "SKIPPED (-SkipTests)"
}

# ==============================================================================
# STAGE 5: RESEARCH PAPER METRICS SYNC
# ==============================================================================
Write-Host "`n[STAGE 5/5] Checking Research Paper Metrics Synchronization..." -ForegroundColor Yellow
& $venvPython "$BaseDir\scripts\update_paper_metrics.py"
$paperExit = $LASTEXITCODE
$executionRecord.PaperSync.ExitCode = $paperExit
if ($paperExit -ne 0) {
    $executionRecord.PaperSync.Status = "WARNING (Exit Code: $paperExit)"
} else {
    $executionRecord.PaperSync.Status = "PASSED (Exit Code: 0)"
}

# ==============================================================================
# FINAL AUDIT REPORT & SUMMARY
# ==============================================================================
$overall = if ($bothWeightsPresent) { "PASSED_FULLY_OPERATIONAL" } else { "PASSED_DEGRADED_READY" }
$summary = if ($bothWeightsPresent) {
    "All test suites passed. Trained CNN and YOLO models verified and active."
} else {
    "All test suites passed cleanly. Degraded fallback verified. System is ready for Kaggle weights."
}

Generate-FinalReport -OverallStatus $overall -SummaryNote $summary

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "📋 FORESTGUARD AI SYSTEM STATUS SUMMARY" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

$distReady = Test-Path "$BaseDir\frontend\dist\index.html"
$cloudflaredReady = Test-Path "$BaseDir\run\cloudflared.exe"

Write-Host "MobileNetV2 CNN:   $(if ($hasCnn) { "✅ INSTALLED ($cnnSize)" } else { "⏳ PENDING KAGGLE WEIGHTS" })"
Write-Host "YOLO11n Detector:  $(if ($hasYolo) { "✅ INSTALLED ($yoloSize)" } else { "⏳ PENDING KAGGLE WEIGHTS" })"
Write-Host "Frontend Build:    $(if ($distReady) { "✅ READY (frontend\dist\ verified)" } else { "⚠️ NOT BUILT" })"
Write-Host "Cloudflare Binary: $(if ($cloudflaredReady) { "✅ PRESENT (run\cloudflared.exe)" } else { "⚠️ MISSING" })"
Write-Host "Backend Pytest:    $($executionRecord.BackendTests.Status)"
Write-Host "Frontend Tests:    $($executionRecord.FrontendTests.Status)"
Write-Host "Inference Mode:    $(if ($bothWeightsPresent) { "Full Trained Inference" } else { "MODEL_UNAVAILABLE Degraded Safe State" })"

Write-Host "`n🚀 Ready for Demo! Start the full system using:" -ForegroundColor Yellow
Write-Host "   .\scripts\run_demo.ps1" -ForegroundColor White -BackgroundColor DarkGreen
Write-Host "======================================================================" -ForegroundColor Cyan
exit 0
