# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for ClipFinder

import sys
from pathlib import Path

block_cipher = None

# Whisperのモデルアセットを含める
import whisper
whisper_path = Path(whisper.__file__).parent
whisper_assets = [(str(whisper_path / "assets"), "whisper/assets")]

# FFmpegバイナリ（存在する場合のみ）
ffmpeg_binaries = []
ffmpeg_dir = Path("ffmpeg")
if (ffmpeg_dir / "ffmpeg.exe").exists():
    ffmpeg_binaries.append((str(ffmpeg_dir / "ffmpeg.exe"), "ffmpeg"))
if (ffmpeg_dir / "ffprobe.exe").exists():
    ffmpeg_binaries.append((str(ffmpeg_dir / "ffprobe.exe"), "ffmpeg"))

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=ffmpeg_binaries,
    datas=whisper_assets,
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'whisper',
        'torch',
        'torchaudio',
        'ffmpeg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ClipFinder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUIアプリなのでコンソール非表示
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # アイコンがあれば 'icon.ico' を指定
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ClipFinder',
)
