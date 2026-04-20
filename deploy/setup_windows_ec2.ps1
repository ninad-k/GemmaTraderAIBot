# ═══════════════════════════════════════════════════════════
#  Rey Capital AI Bot — Windows EC2 Setup Script
# ═══════════════════════════════════════════════════════════
# Run on a fresh Windows Server EC2 instance (g4dn.xlarge recommended for GPU)
# PowerShell: Set-ExecutionPolicy Bypass -Scope Process -Force; .\setup_windows_ec2.ps1

$ErrorActionPreference = "Stop"
$BOT_DIR = "C:\ReyCapital"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Rey Capital AI Bot — EC2 Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ─── 1. Install Python 3.11 ──────────────────────────────
Write-Host "`n[1/6] Installing Python 3.11..." -ForegroundColor Yellow
$pythonInstaller = "$env:TEMP\python-3.11.9-amd64.exe"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile $pythonInstaller
    Start-Process -Wait -FilePath $pythonInstaller -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_pip=1"
    $env:PATH = "C:\Program Files\Python311;C:\Program Files\Python311\Scripts;" + $env:PATH
    Write-Host "  Python installed." -ForegroundColor Green
} else {
    Write-Host "  Python already installed." -ForegroundColor Green
}

# ─── 2. Install Ollama ───────────────────────────────────
Write-Host "`n[2/6] Installing Ollama..." -ForegroundColor Yellow
$ollamaInstaller = "$env:TEMP\OllamaSetup.exe"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $ollamaInstaller
    Start-Process -Wait -FilePath $ollamaInstaller -ArgumentList "/S"
    $env:PATH = "$env:LOCALAPPDATA\Programs\Ollama;" + $env:PATH
    Write-Host "  Ollama installed." -ForegroundColor Green
} else {
    Write-Host "  Ollama already installed." -ForegroundColor Green
}

# ─── 3. Pull Gemma 4 Model ───────────────────────────────
Write-Host "`n[3/6] Pulling Gemma 4 model (this may take a while)..." -ForegroundColor Yellow
Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow
Start-Sleep -Seconds 5
& ollama pull gemma4
Write-Host "  Gemma 4 model ready." -ForegroundColor Green

# ─── 4. Install MetaTrader 5 ─────────────────────────────
Write-Host "`n[4/6] Installing MetaTrader 5..." -ForegroundColor Yellow
$mt5Installer = "$env:TEMP\mt5setup.exe"
if (-not (Test-Path "C:\Program Files\MetaTrader 5\terminal64.exe")) {
    Invoke-WebRequest -Uri "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" -OutFile $mt5Installer
    Start-Process -Wait -FilePath $mt5Installer -ArgumentList "/auto"
    Write-Host "  MetaTrader 5 installed." -ForegroundColor Green
    Write-Host "  NOTE: You must login to MT5 manually with your broker credentials." -ForegroundColor Yellow
} else {
    Write-Host "  MetaTrader 5 already installed." -ForegroundColor Green
}

# ─── 5. Setup Bot ────────────────────────────────────────
Write-Host "`n[5/6] Setting up Rey Capital AI Bot..." -ForegroundColor Yellow
if (-not (Test-Path $BOT_DIR)) {
    New-Item -ItemType Directory -Path $BOT_DIR -Force | Out-Null
}

# Create virtual environment
python -m venv "$BOT_DIR\venv"
& "$BOT_DIR\venv\Scripts\activate.ps1"

# Install dependencies
pip install --upgrade pip
pip install flask flask-socketio pyyaml requests pandas pandas-ta numpy schedule eventlet MetaTrader5

Write-Host "  Dependencies installed." -ForegroundColor Green
Write-Host "  Copy your bot files to $BOT_DIR" -ForegroundColor Yellow

# ─── 6. Firewall Rule ────────────────────────────────────
Write-Host "`n[6/6] Configuring firewall..." -ForegroundColor Yellow
New-NetFirewallRule -DisplayName "Rey Capital Dashboard" -Direction Inbound -Protocol TCP -LocalPort 8050 -Action Allow -ErrorAction SilentlyContinue
Write-Host "  Port 8050 opened." -ForegroundColor Green

# ─── 7. Create Windows Service (optional) ────────────────
Write-Host "`n[Optional] To run as a Windows Service:" -ForegroundColor Cyan
Write-Host "  1. Download NSSM: https://nssm.cc/download" -ForegroundColor White
Write-Host "  2. nssm install ReyCapitalBot `"$BOT_DIR\venv\Scripts\python.exe`" `"$BOT_DIR\run.py`"" -ForegroundColor White
Write-Host "  3. nssm set ReyCapitalBot AppDirectory $BOT_DIR" -ForegroundColor White
Write-Host "  4. nssm start ReyCapitalBot" -ForegroundColor White

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Copy bot files to $BOT_DIR" -ForegroundColor White
Write-Host "  2. Edit config.yaml with your MT5 credentials" -ForegroundColor White
Write-Host "  3. Start MT5 desktop and login to your broker" -ForegroundColor White
Write-Host "  4. Run: cd $BOT_DIR && venv\Scripts\activate && python run.py" -ForegroundColor White
Write-Host "  5. Open: http://localhost:8050" -ForegroundColor White
