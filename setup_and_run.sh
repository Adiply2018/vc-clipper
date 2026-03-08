#!/bin/bash
# ClipFinder セットアップ & 起動スクリプト (Linux/Ubuntu)
# 初回: ./setup_and_run.sh → 2回目以降: ./run.sh または python app.py

echo "========================================"
echo " ClipFinder セットアップ"
echo "========================================"

# Python確認
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 が見つかりません。"
    echo "sudo apt install python3 python3-venv でインストールしてください。"
    exit 1
fi

# ffmpeg確認
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARN] ffmpeg が見つかりません。"
    echo "sudo apt install ffmpeg でインストールしてください。"
    read -p "続行しますか? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# uv確認・仮想環境作成
if command -v uv &> /dev/null; then
    echo "uvを使用して環境をセットアップ中..."
    if [ ! -d ".venv" ]; then
        uv venv
    fi
    source .venv/bin/activate
    uv pip install PyQt6 faster-whisper ffmpeg-python torch
else
    echo "pipを使用して環境をセットアップ中..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install --upgrade pip -q
    pip install PyQt6 faster-whisper ffmpeg-python torch -q
fi

echo ""
echo "セットアップ完了。アプリを起動します..."
python app.py
