@echo off
echo ========================================
echo EYE CLASSIFICATION TRAINING SCRIPT
echo ========================================
echo.
echo Checking Python version...
python --version

echo.
echo Checking TensorFlow installation...
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__} ready!')" 2>nul
if errorlevel 1 (
    echo ❌ TensorFlow not found!
    echo Please run setup.bat first to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting model training...
echo This may take 5-15 minutes depending on your hardware.
echo.
python eye_classifier.py

echo.
echo Training completed!
echo.
echo To test the model:
echo python inference.py --image path/to/eye.jpg
echo.
echo For webcam detection:
echo python inference.py --webcam
echo.
pause