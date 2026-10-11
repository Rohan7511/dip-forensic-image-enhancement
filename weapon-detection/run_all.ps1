# run_all.ps1
# ===========
# PowerShell script to reproduce all phases of the weapon-detection project.
# Run from the weapon-detection/ directory with the venv activated.
#
# Usage:
#   cd weapon-detection
#   .venv\Scripts\Activate.ps1
#   .\run_all.ps1                  # Run all phases
#   .\run_all.ps1 -Phase 1         # Run only a specific phase
#   .\run_all.ps1 -SkipPhases 4    # Skip phase 4 (experiments)
#   .\run_all.ps1 -SmokeOnly       # Only run smoke test (no full training)

param(
    [string]$Phase = "all",
    [string[]]$SkipPhases = @(),
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

# Ensure UTF-8 output so Unicode chars print correctly on Windows
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

function Write-Header($msg) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host ""
}

function Run-Phase($phaseNum, $phaseTitle, $scriptBlock) {
    if ($Phase -ne "all" -and $Phase -ne $phaseNum.ToString()) { return }
    if ($SkipPhases -contains $phaseNum.ToString()) {
        Write-Host "  [SKIP] Phase $phaseNum — $phaseTitle" -ForegroundColor Yellow
        return
    }
    Write-Header "Phase $phaseNum — $phaseTitle"
    & $scriptBlock
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAILED] Phase $phaseNum exited with code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host ""
    Write-Host "  ✓ Phase $phaseNum complete." -ForegroundColor Green
}

# ──────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT: install dependencies
# ──────────────────────────────────────────────────────────────────────────────
Write-Header "Installing dependencies"
python -m pip install --upgrade pip -q
python -m pip install -r "$Root\requirements.txt" -q
Write-Host "  ✓ Dependencies installed." -ForegroundColor Green

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 0: Check dataset
# ──────────────────────────────────────────────────────────────────────────────
Write-Header "Phase 0 — Dataset check"
$dataRaw = "$Root\data\raw"
if (-not (Test-Path $dataRaw) -or (Get-ChildItem $dataRaw -Recurse -File | Measure-Object).Count -eq 0) {
    Write-Host @"

  ⚠  Dataset not found in data/raw/.

  Option A (Kaggle API):
    1. Place your kaggle.json at: C:\Users\<USERNAME>\.kaggle\kaggle.json
    2. Run: python -c "import kagglehub; kagglehub.dataset_download('raghavnanjappan/weapon-dataset-for-yolov5', path='data/raw')"

  Option B (manual download):
    1. Download from: https://www.kaggle.com/datasets/raghavnanjappan/weapon-dataset-for-yolov5
    2. Unzip into: weapon-detection\data\raw\
    3. Re-run this script.

"@ -ForegroundColor Yellow
    exit 1
}
Write-Host "  ✓ Dataset found in data/raw/." -ForegroundColor Green

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1: Data audit and split
# ──────────────────────────────────────────────────────────────────────────────
Run-Phase 1 "Data audit, dedup, split, EDA" {
    python "$Root\scripts\audit_split.py"
}

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2: Training
# ──────────────────────────────────────────────────────────────────────────────
Run-Phase 2 "Training YOLOv5s" {
    if ($SmokeOnly) {
        Write-Host "  [SmokeOnly] Running smoke test only." -ForegroundColor Yellow
        python "$Root\scripts\train.py" --smoke
    } else {
        python "$Root\scripts\train.py"
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3: Evaluation
# ──────────────────────────────────────────────────────────────────────────────
Run-Phase 3 "Evaluation" {
    python "$Root\scripts\evaluate.py"
}

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4: Experiments
# ──────────────────────────────────────────────────────────────────────────────
Run-Phase 4 "Experiments (model comparison + CLAHE ablation)" {
    python "$Root\scripts\experiments.py"
}

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5: Architecture inspection
# ──────────────────────────────────────────────────────────────────────────────
Run-Phase 5 "Architecture inspection" {
    python "$Root\scripts\inspect_model.py"
}

# ──────────────────────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────────────────────
Write-Header "All phases complete"
Write-Host "  Results : $Root\results\" -ForegroundColor Green
Write-Host "  Metrics : $Root\results\metrics.md" -ForegroundColor Green
Write-Host ""
Write-Host "  To run the demo app:" -ForegroundColor Cyan
Write-Host "    python $Root\app.py" -ForegroundColor White
Write-Host ""
Write-Host "  To run inference on an image:" -ForegroundColor Cyan
Write-Host "    python $Root\scripts\predict.py --source <image_path>" -ForegroundColor White
