# ============================================================
#  start.ps1 - Launch NeuroSeg AI (Backend + Frontend)
#  Run from the project root: .\start.ps1
# ============================================================

$Root     = $PSScriptRoot
$Frontend = Join-Path $Root "frontend"

Write-Host ""
Write-Host "  Starting NeuroSeg AI..." -ForegroundColor Cyan
Write-Host ""

# -- Frontend Build -------------------------------------------
Write-Host "  [1/2] Building Production Frontend..." -ForegroundColor Yellow
Set-Location $Frontend
npm run build
Set-Location $Root

# -- Backend --------------------------------------------------
Write-Host "  [2/2] Launching Backend Server -> http://localhost:8000" -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root'; Write-Host 'Backend - Serving NeuroSeg AI' -ForegroundColor Cyan; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000" -WindowStyle Normal

# -- Done -----------------------------------------------------
Write-Host ""
Write-Host "  ✅ Project built and serving from Backend." -ForegroundColor Green
Write-Host ""
Write-Host "  Full App  ->  http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs  ->  http://localhost:8000/docs" -ForegroundColor White
Write-Host ""

# Open browser after a short delay
Start-Sleep -Seconds 5
Start-Process "http://localhost:8000"
