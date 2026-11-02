@echo off
REM ================================================================
REM  PhishGuard BERT — Windows Setup Script
REM  Run this ONCE on Day 1 as Administrator
REM ================================================================

echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Download from python.org/downloads
    pause
    exit /b 1
)

echo [2/5] Upgrading pip...
python -m pip install --upgrade pip

echo [3/5] Installing PyTorch with CUDA support...
REM Installs GPU-accelerated PyTorch (CUDA 11.8)
REM If your GPU is newer, visit pytorch.org to get the right command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo [4/5] Installing ML and API libraries...
pip install transformers==4.40.0
pip install datasets==2.19.0
pip install scikit-learn==1.4.2
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install flask==3.0.3
pip install flask-cors==4.0.1
pip install matplotlib==3.9.0
pip install seaborn==0.13.2
pip install tqdm==4.66.4
pip install accelerate==0.30.0

echo [5/5] Verifying GPU is detected...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo ================================================================
echo  Setup complete! Check above that CUDA available = True
echo  If False, visit: pytorch.org/get-started/locally
echo ================================================================
pause
