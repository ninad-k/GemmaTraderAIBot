@echo off
REM ═══════════════════════════════════════════════════════
REM  Rey Capital AI Bot — Local Windows Setup
REM ═══════════════════════════════════════════════════════
REM  Run this from the gemma_trader directory.

echo.
echo ========================================
echo   Rey Capital AI Bot - Local Setup
echo ========================================
echo.

REM ─── Check Python ──────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Install from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found.

REM ─── Create venv ───────────────────────────────────────
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
)
echo [OK] Virtual environment ready.

REM ─── Activate and install deps ─────────────────────────
echo [2/4] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
pip install MetaTrader5

REM ─── Check Ollama ──────────────────────────────────────
echo [3/4] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not found!
    echo   Download from: https://ollama.com/download
    echo   Then run: ollama pull gemma4
) else (
    echo [OK] Ollama found.
    echo   Pulling Gemma 4 model...
    ollama pull gemma4
)

REM ─── Check MT5 ─────────────────────────────────────────
echo [4/4] Checking MetaTrader 5...
if exist "C:\Program Files\MetaTrader 5\terminal64.exe" (
    echo [OK] MetaTrader 5 found.
) else (
    echo [WARNING] MetaTrader 5 not found at default path.
    echo   Make sure MT5 is installed and running.
)

REM ─── Create logs dir ───────────────────────────────────
if not exist "logs" mkdir logs

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To start the bot:
echo   1. Open MetaTrader 5 and login to your broker
echo   2. Make sure Ollama is running: ollama serve
echo   3. Run: venv\Scripts\activate ^& python run.py
echo   4. Open: http://localhost:8050
echo.
pause
