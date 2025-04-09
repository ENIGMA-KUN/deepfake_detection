@echo off
echo ===================================
echo DeepFake Detective Setup and Launch
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the PATH.
    echo Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create necessary directories
echo Setting up directories...
if not exist assets\sample_images\real mkdir assets\sample_images\real
if not exist assets\sample_images\fake mkdir assets\sample_images\fake
if not exist models mkdir models

REM Check if model exists
if not exist models\xception.pth (
    echo.
    echo WARNING: Model file not found at models\xception.pth
    echo Please ensure you have placed the xception.pth file in the models directory.
    echo.
    echo Press any key to continue anyway...
    pause >nul
)

REM Check if sample images exist
set /a real_count=0
for %%f in (assets\sample_images\real\*.*) do set /a real_count+=1

set /a fake_count=0
for %%f in (assets\sample_images\fake\*.*) do set /a fake_count+=1

if %real_count% lss 10 (
    echo.
    echo WARNING: Not enough real sample images found (%real_count%/10 minimum)
    echo Please add more real face images to assets\sample_images\real
    echo.
)

if %fake_count% lss 10 (
    echo.
    echo WARNING: Not enough fake sample images found (%fake_count%/10 minimum)
    echo Please add more fake face images to assets\sample_images\fake
    echo.
)

if %real_count% lss 10 (
    echo Press any key to continue anyway...
    pause >nul
)

REM Run the application
echo.
echo Starting DeepFake Detective...
streamlit run app.py

REM Deactivate virtual environment when the app is closed
call venv\Scripts\deactivate.bat