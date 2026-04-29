@echo off
echo ========================================
echo EYE CLASSIFICATION SETUP SCRIPT
echo ========================================
echo.
echo This script will help you set up the environment for eye classification.
echo.
echo Current Python version: Checking...
python --version
echo.
echo IMPORTANT: TensorFlow requires Python 3.11 or earlier.
echo You currently have Python 3.14 which is not compatible.
echo.
echo STEP 1: Install Python 3.11
echo ================================
echo.
echo Option A - Download from python.org:
echo 1. Go to https://www.python.org/downloads/release/python-3118/
echo 2. Download "Windows installer (64-bit)"
echo 3. Install Python 3.11 to a folder like C:\Python311
echo 4. Make sure to check "Add Python to PATH" during installation
echo.
echo Option B - Use Microsoft Store:
echo 1. Open Microsoft Store
echo 2. Search for "Python 3.11"
echo 3. Install it
echo.
echo After installing Python 3.11, run this script again.
echo.
pause
goto :setup_python311

:setup_python311
echo.
echo STEP 2: Verify Python 3.11 Installation
echo ========================================
echo.
python --version | findstr "3.11" >nul
if errorlevel 1 (
    echo ERROR: Python 3.11 not found in PATH.
    echo Please make sure Python 3.11 is installed and added to PATH.
    echo.
    echo To check your Python installations:
    echo where python
    echo.
    echo You may need to use the full path, e.g.:
    echo C:\Python311\python.exe --version
    echo.
    pause
    exit /b 1
) else (
    echo ✓ Python 3.11 found!
)

echo.
echo STEP 3: Install Dependencies
echo ============================
echo.
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install tensorflow opencv-python numpy matplotlib scikit-learn pillow

if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ✓ All dependencies installed successfully!
echo.
echo STEP 4: Verify Installation
echo ============================
echo.
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__} installed')"
python -c "import cv2; print('✓ OpenCV installed')"
python -c "import numpy as np; print('✓ NumPy installed')"
python -c "import matplotlib; print('✓ Matplotlib installed')"

echo.
echo 🎉 SETUP COMPLETE!
echo =================
echo.
echo You can now train the model by running:
echo python eye_classifier.py
echo.
echo Or test inference with:
echo python inference.py --image path/to/eye.jpg
echo python inference.py --webcam
echo.
pause