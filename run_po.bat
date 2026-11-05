@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
  echo [1/3] Creating Python virtual environment...
  py -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment. Install Python 3.10+ and try again.
    pause
    exit /b 1
  )
)

echo [2/3] Installing required packages...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Check your internet connection or company network restrictions.
  pause
  exit /b 1
)

echo [3/3] Starting PO Extract...
echo Keep this window open while using the app.
.venv\Scripts\python.exe -m streamlit run po_app.py
pause
