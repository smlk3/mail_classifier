@echo off
setlocal
echo ===================================================
echo Mistral Mail Asistani - Kurulum ve Baslatma
echo ===================================================
echo.
echo [BILGI] Bu islem internet baglantisi gerektirir.
echo.

echo 1. PyTorch (GPU Destekli) kontrol ediliyor...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo 2. Gereksinimler yukleniyor...
pip install -r requirements.txt

echo.
echo 3. Uygulama baslatiliyor...
echo [NOT] Ilk acilista model indirilecegi icin (yaklasik 4-5 GB) beklemeniz gerekebilir.
echo.
streamlit run app.py
pause
