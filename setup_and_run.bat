@echo off
REM ClipFinder セットアップ & 起動スクリプト (Windows)
REM 初回: setup.bat → 2回目以降: run.bat または python app.py

echo ========================================
echo  ClipFinder セットアップ
echo ========================================

REM Python確認
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python が見つかりません。
    echo https://www.python.org からインストールしてください。
    pause
    exit /b 1
)

REM ffmpeg確認
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARN] ffmpeg が見つかりません。
    echo https://ffmpeg.org/download.html からインストールし、
    echo PATHに追加してください。
    pause
)

REM 仮想環境作成
if not exist venv (
    echo 仮想環境を作成中...
    python -m venv venv
)

REM 依存パッケージインストール
echo パッケージをインストール中...
call venv\Scripts\activate
pip install --upgrade pip -q
pip install PyQt6 faster-whisper ffmpeg-python -q

REM CUDA確認してtorchインストール
python -c "import torch; print(torch.cuda.is_available())" >nul 2>&1
if errorlevel 1 (
    REM PyTorchまだない場合
    echo PyTorch をインストール中...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    if errorlevel 1 (
        pip install torch -q
    )
)

echo.
echo セットアップ完了。アプリを起動します...
python app.py
