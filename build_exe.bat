@echo off
REM ClipFinder Windows実行ファイルビルドスクリプト
REM 必要: Python 3.10+, pip

echo ========================================
echo  ClipFinder EXEビルド
echo ========================================

REM 仮想環境作成
if not exist build_venv (
    echo 仮想環境を作成中...
    python -m venv build_venv
)

call build_venv\Scripts\activate

echo 依存パッケージをインストール中...
pip install --upgrade pip
pip install pyinstaller PyQt6 openai-whisper ffmpeg-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM FFmpegダウンロード
if not exist ffmpeg (
    echo FFmpegをダウンロード中...
    mkdir ffmpeg
    curl -L -o ffmpeg.zip https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
    tar -xf ffmpeg.zip
    copy ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe ffmpeg\
    copy ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe ffmpeg\
    rmdir /s /q ffmpeg-master-latest-win64-gpl
    del ffmpeg.zip
)

echo PyInstallerでビルド中...
pyinstaller --noconfirm --clean clipfinder.spec

echo.
echo ========================================
echo  ビルド完了!
echo  出力: dist\ClipFinder\
echo ========================================
pause
