@echo off
REM SAM2 to SAM3 Migration Script
REM ==============================
REM This script upgrades PyTorch and installs SAM3 dependencies

echo ============================================================
echo SAM2 to SAM3 Migration Script
echo ============================================================
echo.

REM Activate conda environment
call conda activate autoroto
if %ERRORLEVEL% neq 0 (
    echo ERROR: Could not activate autoroto conda environment
    echo Please ensure conda is initialized and autoroto env exists
    pause
    exit /b 1
)

echo Current environment:
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')"
echo.

echo ============================================================
echo Phase 1: Upgrading PyTorch to 2.7+ with CUDA 12.8
echo ============================================================
echo.

REM Uninstall current PyTorch
echo Uninstalling current PyTorch...
pip uninstall torch torchvision torchaudio -y

REM Install PyTorch 2.7 with CUDA 12.8
echo Installing PyTorch 2.7 with CUDA 12.8...
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install PyTorch 2.7
    echo Attempting fallback to cu126...
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
)

echo.
echo ============================================================
echo Phase 2: Installing triton-windows for torch.compile support
echo ============================================================
echo.

pip install -U "triton-windows>=3.3,<3.4"

echo.
echo ============================================================
echo Phase 3: Installing Ultralytics with SAM3 support
echo ============================================================
echo.

pip install -U "ultralytics>=8.3.237"

echo.
echo ============================================================
echo Phase 4: Clearing old caches
echo ============================================================
echo.

if exist "%USERPROFILE%\.triton\cache" (
    echo Clearing Triton cache...
    rmdir /s /q "%USERPROFILE%\.triton\cache"
)

if exist "%LOCALAPPDATA%\Temp\torchinductor_%USERNAME%" (
    echo Clearing TorchInductor cache...
    rmdir /s /q "%LOCALAPPDATA%\Temp\torchinductor_%USERNAME%"
)

echo.
echo ============================================================
echo Verification
echo ============================================================
echo.

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "from ultralytics import __version__; print(f'Ultralytics: {__version__}')"

echo.
echo ============================================================
echo Migration complete!
echo ============================================================
echo.
echo IMPORTANT: SAM3 weights (sam3.pt) must be downloaded manually:
echo   1. Request access at https://huggingface.co/facebook/sam3
echo   2. Download sam3.pt after approval
echo   3. Place sam3.pt in your working directory
echo.
echo To test SAM3:
echo   python -c "from ultralytics.models.sam import SAM3SemanticPredictor; print('SAM3 available')"
echo.
pause
