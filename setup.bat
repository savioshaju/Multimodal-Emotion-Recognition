@echo off
setlocal

:: Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo Detected Python version: %PYTHON_VERSION%

echo %PYTHON_VERSION% | findstr /b /c:"3.11." >nul
if errorlevel 1 (
    echo Error: Python version must be 3.11.x.
    echo Current version is %PYTHON_VERSION%.
    exit /b 1
)

:: Create venv
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
if exist "requirements.txt" (
    echo Installing requirements...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found!
)

echo Setup completed successfully!
endlocal
