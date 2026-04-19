# ================================================================
#  Rey Capital AI Bot -- Windows VPS Deployment Script
# ================================================================
#
#  Run this on a fresh Windows Server VPS (2019/2022) via RDP:
#    1. Open PowerShell as Administrator
#    2. Set-ExecutionPolicy Bypass -Scope Process -Force
#    3. .\vps_setup.ps1
#
#  Recommended VPS specs:
#    - 4+ CPU cores, 16+ GB RAM
#    - 100 GB SSD
#    - GPU optional (NVIDIA T4/RTX for faster Gemma inference)
#    - Windows Server 2019 or 2022
#
# ================================================================

param(
    [string]$BotDir = "C:\ReyCapital",
    [string]$PythonVersion = "3.12.8",
    [switch]$SkipMT5,
    [switch]$SkipOllama,
    [switch]$ServiceOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Speed up downloads

function Write-Step($num, $total, $msg) {
    Write-Host "`n[$num/$total] $msg" -ForegroundColor Yellow
}

function Write-OK($msg) {
    Write-Host "  [OK] $msg" -ForegroundColor Green
}

function Write-Warn($msg) {
    Write-Host "  [!] $msg" -ForegroundColor Red
}

function Write-Info($msg) {
    Write-Host "  $msg" -ForegroundColor White
}

$TOTAL_STEPS = 10

Write-Host ""
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host "     Rey Capital AI Bot -- VPS Deployment" -ForegroundColor Cyan
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host "  Target directory: $BotDir" -ForegroundColor White
Write-Host ""

# ================================================================
#  STEP 1: System Prerequisites
# ================================================================
Write-Step 1 $TOTAL_STEPS "Installing system prerequisites..."

# Install Git if not present
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Info "Downloading Git..."
    $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/Git-2.47.1.2-64-bit.exe"
    $gitInstaller = "$env:TEMP\GitSetup.exe"
    Invoke-WebRequest -Uri $gitUrl -OutFile $gitInstaller
    Start-Process -Wait -FilePath $gitInstaller -ArgumentList "/VERYSILENT", "/NORESTART"
    $env:PATH = "C:\Program Files\Git\cmd;" + $env:PATH
    Write-OK "Git installed"
} else {
    Write-OK "Git already installed"
}

# Install Visual C++ Redistributable (needed by MT5 and some Python packages)
Write-Info "Ensuring Visual C++ Redistributable..."
$vcUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
$vcInstaller = "$env:TEMP\vc_redist.x64.exe"
Invoke-WebRequest -Uri $vcUrl -OutFile $vcInstaller -ErrorAction SilentlyContinue
Start-Process -Wait -FilePath $vcInstaller -ArgumentList "/quiet", "/norestart" -ErrorAction SilentlyContinue
Write-OK "Visual C++ Redistributable OK"

# ================================================================
#  STEP 2: Install Python
# ================================================================
Write-Step 2 $TOTAL_STEPS "Installing Python $PythonVersion..."

$pythonCheck = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCheck) {
    $pyMajorMinor = $PythonVersion.Substring(0, $PythonVersion.LastIndexOf('.'))
    $pyUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
    $pyInstaller = "$env:TEMP\python-$PythonVersion-amd64.exe"
    Write-Info "Downloading Python $PythonVersion..."
    Invoke-WebRequest -Uri $pyUrl -OutFile $pyInstaller
    Write-Info "Installing Python (this may take a minute)..."
    Start-Process -Wait -FilePath $pyInstaller -ArgumentList `
        "/quiet", "InstallAllUsers=1", "PrependPath=1", `
        "Include_pip=1", "Include_launcher=1"
    # Refresh PATH
    $env:PATH = "C:\Program Files\Python312;C:\Program Files\Python312\Scripts;" + $env:PATH
    Write-OK "Python $PythonVersion installed"
} else {
    $pyVer = python --version 2>&1
    Write-OK "Python already installed: $pyVer"
}

# Verify pip
python -m pip install --upgrade pip 2>$null
Write-OK "pip upgraded"

# ================================================================
#  STEP 3: Install Ollama + Gemma 4
# ================================================================
if (-not $SkipOllama) {
    Write-Step 3 $TOTAL_STEPS "Installing Ollama + Gemma 4 AI model..."

    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Info "Downloading Ollama..."
        $ollamaUrl = "https://ollama.com/download/OllamaSetup.exe"
        $ollamaInstaller = "$env:TEMP\OllamaSetup.exe"
        Invoke-WebRequest -Uri $ollamaUrl -OutFile $ollamaInstaller
        Write-Info "Installing Ollama..."
        Start-Process -Wait -FilePath $ollamaInstaller -ArgumentList "/S"
        $env:PATH = "$env:LOCALAPPDATA\Programs\Ollama;" + $env:PATH
        # Wait for Ollama service to start
        Start-Sleep -Seconds 5
        Write-OK "Ollama installed"
    } else {
        Write-OK "Ollama already installed"
    }

    # Ensure Ollama is running
    Write-Info "Starting Ollama service..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3

    # Pull Gemma 4 model
    Write-Info "Pulling Gemma 4 model (~5GB download, be patient)..."
    & ollama pull gemma4
    Write-OK "Gemma 4 model ready"
} else {
    Write-Step 3 $TOTAL_STEPS "Skipping Ollama (--SkipOllama)"
}

# ================================================================
#  STEP 4: Install MetaTrader 5
# ================================================================
if (-not $SkipMT5) {
    Write-Step 4 $TOTAL_STEPS "Installing MetaTrader 5..."

    $mt5Paths = @(
        "C:\Program Files\MetaTrader 5\terminal64.exe",
        "C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        "$env:APPDATA\MetaQuotes\Terminal\*\terminal64.exe"
    )
    $mt5Found = $false
    foreach ($p in $mt5Paths) {
        if (Test-Path $p) { $mt5Found = $true; break }
    }

    if (-not $mt5Found) {
        $mt5Url = "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
        $mt5Installer = "$env:TEMP\mt5setup.exe"
        Write-Info "Downloading MetaTrader 5..."
        Invoke-WebRequest -Uri $mt5Url -OutFile $mt5Installer
        Write-Info "Installing MetaTrader 5..."
        Start-Process -Wait -FilePath $mt5Installer -ArgumentList "/auto"
        Write-OK "MetaTrader 5 installed"
    } else {
        Write-OK "MetaTrader 5 already installed"
    }

    Write-Warn "MANUAL STEP: Open MT5, login to your broker, enable AutoTrading (green button)"
} else {
    Write-Step 4 $TOTAL_STEPS "Skipping MT5 (--SkipMT5)"
}

# ================================================================
#  STEP 5: Clone/Copy Bot Files
# ================================================================
Write-Step 5 $TOTAL_STEPS "Setting up bot directory at $BotDir..."

if (-not (Test-Path $BotDir)) {
    New-Item -ItemType Directory -Path $BotDir -Force | Out-Null
}

# If git repo URL is available, clone it
$repoUrl = "https://github.com/ninad-k/AlgoStrategies.git"
$gemmaTraderSrc = "$BotDir\AlgoStrategies\execution\gemma_trader"

if (-not (Test-Path "$BotDir\AlgoStrategies")) {
    Write-Info "Cloning repository..."
    git clone $repoUrl "$BotDir\AlgoStrategies"
    Write-OK "Repository cloned"
} else {
    Write-Info "Pulling latest changes..."
    Push-Location "$BotDir\AlgoStrategies"
    git pull origin main
    Pop-Location
    Write-OK "Repository updated"
}

# Create working directory symlink / copy
$workDir = "$BotDir\bot"
if (-not (Test-Path $workDir)) {
    Copy-Item -Path $gemmaTraderSrc -Destination $workDir -Recurse
    Write-OK "Bot files copied to $workDir"
} else {
    Write-OK "Bot directory already exists at $workDir"
}

# ================================================================
#  STEP 6: Python Virtual Environment + Dependencies
# ================================================================
Write-Step 6 $TOTAL_STEPS "Creating Python virtual environment and installing dependencies..."

$venvDir = "$BotDir\venv"
if (-not (Test-Path "$venvDir\Scripts\python.exe")) {
    python -m venv $venvDir
    Write-OK "Virtual environment created"
}

# Activate and install
& "$venvDir\Scripts\python.exe" -m pip install --upgrade pip
& "$venvDir\Scripts\pip.exe" install `
    flask flask-socketio pyyaml requests `
    pandas pandas_ta numpy schedule eventlet `
    MetaTrader5 tqdm scipy stockstats python-docx
Write-OK "All dependencies installed"

# ================================================================
#  STEP 7: Create Logs Directory + Initialize Files
# ================================================================
Write-Step 7 $TOTAL_STEPS "Initializing log files..."

$logsDir = "$workDir\logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

$jsonFiles = @(
    "gemma_decisions.json",
    "trades.json",
    "trade_outcomes.json",
    "parameter_adjustments.json",
    "trade_journal.json"
)
foreach ($f in $jsonFiles) {
    $fp = "$logsDir\$f"
    if (-not (Test-Path $fp)) {
        "[]" | Out-File -FilePath $fp -Encoding utf8NoBOM
        Write-Info "Created $f"
    }
}

# Lot overrides
$lotPath = "$logsDir\lot_overrides.json"
if (-not (Test-Path $lotPath)) {
    "{}" | Out-File -FilePath $lotPath -Encoding utf8NoBOM
    Write-Info "Created lot_overrides.json"
}
Write-OK "Log files initialized"

# ================================================================
#  STEP 8: Configure Firewall
# ================================================================
Write-Step 8 $TOTAL_STEPS "Configuring Windows Firewall..."

$rules = @(
    @{Name="Rey Capital Dashboard"; Port=8050},
    @{Name="Ollama AI API"; Port=11434}
)

foreach ($rule in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
    if (-not $existing) {
        New-NetFirewallRule -DisplayName $rule.Name `
            -Direction Inbound -Protocol TCP `
            -LocalPort $rule.Port -Action Allow | Out-Null
        Write-OK "Firewall: $($rule.Name) (port $($rule.Port)) opened"
    } else {
        Write-OK "Firewall: $($rule.Name) already configured"
    }
}

# ================================================================
#  STEP 9: Create Startup Script + Windows Service
# ================================================================
Write-Step 9 $TOTAL_STEPS "Creating startup scripts..."

# Batch file for manual start
$batchContent = @"
@echo off
echo =============================================
echo   Rey Capital AI Bot -- Starting...
echo =============================================
echo.

REM Activate virtual environment
call "$venvDir\Scripts\activate.bat"

REM Change to bot directory
cd /d "$workDir"

REM Start the bot
echo Starting Rey Capital AI Bot...
echo Dashboard: http://localhost:8050
echo.
python run.py

pause
"@
$batchContent | Out-File -FilePath "$BotDir\start_bot.bat" -Encoding ascii
Write-OK "Created start_bot.bat"

# PowerShell script for service
$serviceScript = @"
# Rey Capital AI Bot -- Service Runner
`$env:PATH = "C:\Program Files\Python312;C:\Program Files\Python312\Scripts;" + `$env:PATH
Set-Location "$workDir"
& "$venvDir\Scripts\python.exe" run.py 2>&1 | Tee-Object -FilePath "$workDir\logs\service.log" -Append
"@
$serviceScript | Out-File -FilePath "$BotDir\run_service.ps1" -Encoding utf8
Write-OK "Created run_service.ps1"

# Task Scheduler XML
$taskXml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Rey Capital AI Bot -- Automated Crypto Trading</Description>
  </RegistrationInfo>
  <Triggers>
    <BootTrigger>
      <Enabled>true</Enabled>
      <Delay>PT60S</Delay>
    </BootTrigger>
  </Triggers>
  <Principals>
    <Principal>
      <UserId>S-1-5-18</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RestartOnFailure>
      <Interval>PT5M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
  </Settings>
  <Actions>
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-ExecutionPolicy Bypass -File "$BotDir\run_service.ps1"</Arguments>
      <WorkingDirectory>$workDir</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@
$taskXml | Out-File -FilePath "$BotDir\ReyCapitalBot_Task.xml" -Encoding unicode
Write-OK "Created Task Scheduler XML"

# Register the task
Write-Info "Registering scheduled task..."
schtasks /Create /TN "ReyCapitalBot" /XML "$BotDir\ReyCapitalBot_Task.xml" /F 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-OK "Scheduled task 'ReyCapitalBot' registered (starts on boot)"
} else {
    Write-Warn "Could not register task -- run manually: schtasks /Create /TN ReyCapitalBot /XML $BotDir\ReyCapitalBot_Task.xml /F"
}

# ================================================================
#  STEP 10: Verify Installation
# ================================================================
Write-Step 10 $TOTAL_STEPS "Verifying installation..."

$checks = @(
    @{Name="Python"; Cmd="python --version"},
    @{Name="pip"; Cmd="pip --version"},
    @{Name="Git"; Cmd="git --version"},
    @{Name="Flask"; Cmd="$venvDir\Scripts\python.exe -c `"import flask; print(flask.__version__)`""},
    @{Name="pandas_ta"; Cmd="$venvDir\Scripts\python.exe -c `"import pandas_ta; print('OK')`""},
    @{Name="MetaTrader5"; Cmd="$venvDir\Scripts\python.exe -c `"import MetaTrader5; print('OK')`""},
    @{Name="Bot files"; Cmd="if (Test-Path '$workDir\run.py') { 'OK' } else { throw 'missing' }"}
)

foreach ($check in $checks) {
    try {
        $result = Invoke-Expression $check.Cmd 2>&1
        Write-OK "$($check.Name): $result"
    } catch {
        Write-Warn "$($check.Name): FAILED"
    }
}

# Check Ollama
if (-not $SkipOllama) {
    try {
        $ollamaResp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 5
        Write-OK "Ollama: running"
    } catch {
        Write-Warn "Ollama: not responding (start with: ollama serve)"
    }
}

# ================================================================
#  DONE
# ================================================================

Write-Host ""
Write-Host "  =============================================" -ForegroundColor Green
Write-Host "     SETUP COMPLETE!" -ForegroundColor Green
Write-Host "  =============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Files:" -ForegroundColor Cyan
Write-Host "    Bot:       $workDir" -ForegroundColor White
Write-Host "    Venv:      $venvDir" -ForegroundColor White
Write-Host "    Start:     $BotDir\start_bot.bat" -ForegroundColor White
Write-Host "    Config:    $workDir\config.yaml" -ForegroundColor White
Write-Host ""
Write-Host "  BEFORE FIRST RUN - Complete these manual steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "    1. Open MetaTrader 5 and login to your broker" -ForegroundColor White
Write-Host "    2. Click AutoTrading button (must be GREEN)" -ForegroundColor White
Write-Host "    3. In Market Watch: right-click > Show All" -ForegroundColor White
Write-Host "    4. Edit config: notepad $workDir\config.yaml" -ForegroundColor White
Write-Host "       - Set your MT5 login/password (or leave blank if logged in)" -ForegroundColor White
Write-Host "       - Adjust allowed_symbols if needed" -ForegroundColor White
Write-Host ""
Write-Host "  TO START:" -ForegroundColor Cyan
Write-Host "    Double-click: $BotDir\start_bot.bat" -ForegroundColor White
Write-Host "    Or run:       schtasks /Run /TN ReyCapitalBot" -ForegroundColor White
Write-Host ""
Write-Host "  Dashboard: http://localhost:8050" -ForegroundColor Cyan
Write-Host "  Remote:    http://<YOUR-VPS-IP>:8050" -ForegroundColor Cyan
Write-Host ""
