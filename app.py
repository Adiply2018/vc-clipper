"""
ClipFinder - ゲーム動画セリフクリッパー
音声認識でセリフを検索し、前後をクリップするデスクトップアプリ
"""

import sys
import os
from dotenv import load_dotenv

# UTF-8を強制
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('LANG', 'en_US.UTF-8')
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')
import json
import threading
import subprocess
import tempfile
import platform
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# 環境変数の読み込み
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ランダム話者名（SPEAKER_00の代わりにLoLチャンピオン名を使用）
SPEAKER_NAMES = [
    "Ahri", "Jinx", "Lux", "Yasuo", "Zed",
    "Thresh", "Lee Sin", "Ezreal", "Kai'Sa", "Akali",
    "Sett", "Yone", "Viego", "Garen", "Darius",
    "Teemo", "Vayne", "Ashe", "Miss Fortune", "Caitlyn"
]


# ─────────────────────────────────────────────
# 設定管理
# ─────────────────────────────────────────────

class ConfigManager:
    """~/.clipfinder/config.json で設定を管理"""

    def __init__(self):
        self._config_dir = Path.home() / ".clipfinder"
        self._config_file = self._config_dir / "config.json"
        self._config = self._load()

    def _load(self) -> dict:
        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save(self):
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._config_file, "w", encoding="utf-8") as f:
            json.dump(self._config, f, ensure_ascii=False, indent=2)

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    def set(self, key: str, value):
        self._config[key] = value
        self._save()

    @property
    def hf_token(self) -> str:
        return self.get("hf_token", "")

    @hf_token.setter
    def hf_token(self, value: str):
        self.set("hf_token", value)

    @property
    def diarization_enabled(self) -> bool:
        return self.get("diarization_enabled", True)  # デフォルトON

    @diarization_enabled.setter
    def diarization_enabled(self, value: bool):
        self.set("diarization_enabled", value)

    @property
    def speaker_labels(self) -> dict:
        """話者ID -> カスタム名のマッピング"""
        return self.get("speaker_labels", {})

    @speaker_labels.setter
    def speaker_labels(self, value: dict):
        self.set("speaker_labels", value)

    def get_speaker_label(self, speaker_id: str) -> str:
        """話者IDのラベルを取得（カスタム名があればそれを、なければデフォルト）"""
        labels = self.speaker_labels
        return labels.get(speaker_id, speaker_id)

    def set_speaker_label(self, speaker_id: str, label: str):
        """話者IDにカスタムラベルを設定"""
        labels = self.speaker_labels
        labels[speaker_id] = label
        self.speaker_labels = labels


def get_ffmpeg_path() -> str:
    """PyInstallerでパッケージングされた場合は同梱のFFmpegを使用"""
    if getattr(sys, 'frozen', False):
        # PyInstallerでパッケージングされた場合
        base_path = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(sys.executable).parent
        ffmpeg_dir = base_path / "ffmpeg"
        ffmpeg_exe = ffmpeg_dir / ("ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg")
        if ffmpeg_exe.exists():
            return str(ffmpeg_exe)
    return "ffmpeg"  # システムのffmpegを使用

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QProgressBar, QSlider, QSpinBox, QDoubleSpinBox, QFrame,
    QScrollArea, QSplitter, QMessageBox, QComboBox, QCheckBox,
    QGroupBox, QListWidget, QListWidgetItem, QTabWidget,
    QStatusBar, QToolButton, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QSize, QPoint
)
from PyQt6.QtGui import (
    QColor, QPalette, QFont, QFontDatabase, QIcon,
    QPainter, QLinearGradient, QBrush, QPen, QPixmap,
    QDragEnterEvent, QDropEvent
)


# ─────────────────────────────────────────────
# ワーカースレッド: 音声認識
# ─────────────────────────────────────────────

class TranscribeWorker(QThread):
    progress = pyqtSignal(int, str)         # (percent, message)
    segment_ready = pyqtSignal(dict)         # 個別セグメント
    finished = pyqtSignal(list)              # 全セグメント
    error = pyqtSignal(str)

    def __init__(self, video_path: str, model_size: str, language: str = "ja"):
        super().__init__()
        self.video_path = video_path
        self.model_size = model_size
        self.language = language
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from faster_whisper import WhisperModel
            import torch

            self.progress.emit(5, "モデルを読み込み中...")

            # GPU/CPU自動選択
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"  # CPUではint8で高速化

            actual_model = self.model_size
            if self.model_size == "auto":
                actual_model = "large-v3" if torch.cuda.is_available() else "medium"

            self.progress.emit(15, f"Whisper {actual_model} ({device}, {compute_type}) で認識開始...")

            model = WhisperModel(actual_model, device=device, compute_type=compute_type)

            if self._cancelled:
                return

            self.progress.emit(30, "音声を抽出中...")

            # 一時WAVファイルに音声抽出（ASCII文字のみのパスを使用）
            import uuid
            tmp_dir = "/tmp" if platform.system() != "Windows" else tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, f"clipfinder_{uuid.uuid4().hex}.wav")

            cmd = [
                get_ffmpeg_path(), "-y", "-i", self.video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                tmp_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg エラー: {result.stderr}")

            if self._cancelled:
                os.unlink(tmp_path)
                return

            self.progress.emit(50, "音声認識中...")

            # 音声をnumpy配列として読み込んでPyAVのパス問題を回避
            import numpy as np
            import wave
            with wave.open(tmp_path, 'rb') as wf:
                audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0

            os.unlink(tmp_path)  # 早めに削除

            # faster-whisperはジェネレータを返す
            segment_generator, info = model.transcribe(
                audio_float,
                language=self.language,
                word_timestamps=True,
                vad_filter=True,  # 無音部分をスキップして高速化
            )

            segments = []
            for i, seg in enumerate(segment_generator):
                if self._cancelled:
                    break
                segment_data = {
                    "id": i,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
                segments.append(segment_data)
                self.segment_ready.emit(segment_data)
                self.progress.emit(60, f"認識中... {i+1} セグメント")

            if self._cancelled:
                return

            self.progress.emit(100, "完了")
            self.finished.emit(segments)

        except ImportError as e:
            import traceback
            traceback.print_exc()
            self.error.emit(
                "faster-whisper が見つかりません。\n"
                "pip install faster-whisper torch を実行してください。"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ─────────────────────────────────────────────
# ワーカースレッド: 話者分離
# ─────────────────────────────────────────────

class DiarizationWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)  # [(start, end, speaker_id), ...]
    error = pyqtSignal(str)

    # クラス変数でパイプラインをキャッシュ（2回目以降のロードを高速化）
    _pipeline_cache = None
    _pipeline_device = None
    _pipeline_model = None

    def __init__(self, audio_path: str, hf_token: str, min_speakers: int = None, max_speakers: int = None, fast_mode: bool = False):
        super().__init__()
        self.audio_path = audio_path
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.fast_mode = fast_mode
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @classmethod
    def get_pipeline(cls, hf_token: str, device, fast_mode: bool = False):
        """パイプラインをキャッシュから取得（なければ新規作成）"""
        from pyannote.audio import Pipeline

        # 使用するモデルを決定
        model_name = "pyannote/speaker-diarization-3.0" if fast_mode else "pyannote/speaker-diarization-3.1"
        device_str = str(device)

        # キャッシュが有効か確認
        if (cls._pipeline_cache is not None and
            cls._pipeline_device == device_str and
            cls._pipeline_model == model_name):
            return cls._pipeline_cache

        # 新規作成
        pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
        pipeline.to(device)

        # キャッシュに保存
        cls._pipeline_cache = pipeline
        cls._pipeline_device = device_str
        cls._pipeline_model = model_name

        return pipeline

    def run(self):
        try:
            import torch
            from pyannote.audio import Pipeline

            model_name = "3.0（高速）" if self.fast_mode else "3.1（高精度）"
            self.progress.emit(10, f"話者分離: モデル {model_name} を準備中...")

            # HF_TOKEN環境変数を設定（pyannote-audio 3.1+で必要）
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token

            # GPU/CPU自動選択
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # キャッシュからパイプラインを取得
            pipeline = self.get_pipeline(self.hf_token, device, self.fast_mode)

            if self._cancelled:
                return

            self.progress.emit(30, "話者分離: 音声埋め込み計算中... (1/3)")

            # 話者数のヒントを設定
            diarization_params = {}
            if self.min_speakers is not None:
                diarization_params["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                diarization_params["max_speakers"] = self.max_speakers

            self.progress.emit(50, "話者分離: クラスタリング中... (2/3)")

            # 話者分離実行
            diarization = pipeline(self.audio_path, **diarization_params)

            if self._cancelled:
                return

            self.progress.emit(80, "話者分離: ラベル割り当て中... (3/3)")

            # 結果を抽出
            segments = []
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            self.progress.emit(100, "話者分離完了")
            self.finished.emit(segments)

        except ImportError as e:
            import traceback
            traceback.print_exc()
            self.error.emit(
                "pyannote.audio が見つかりません。\n"
                "pip install pyannote.audio を実行してください。"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                self.error.emit(
                    "HuggingFace認証エラー:\n"
                    "1. トークンが正しいか確認してください\n"
                    "2. https://huggingface.co/pyannote/speaker-diarization-3.1 で利用規約に同意してください"
                )
            else:
                self.error.emit(error_msg)


# ─────────────────────────────────────────────
# ワーカースレッド: クリップ出力
# ─────────────────────────────────────────────

class ClipWorker(QThread):
    progress = pyqtSignal(int, str)
    clip_done = pyqtSignal(str)   # 出力ファイルパス
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, jobs: list):
        """
        jobs: [{"video": str, "start": float, "end": float, "output": str}, ...]
        """
        super().__init__()
        self.jobs = jobs

    def run(self):
        try:
            total = len(self.jobs)
            for i, job in enumerate(self.jobs):
                self.progress.emit(
                    int((i + 1) / total * 100),
                    f"クリップ {i+1}/{total} を出力中..."
                )
                duration = job["end"] - job["start"]
                cmd = [
                    get_ffmpeg_path(), "-y",
                    "-ss", str(job["start"]),
                    "-i", job["video"],
                    "-t", str(duration),
                    "-c", "copy",
                    job["output"]
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                if result.returncode != 0:
                    self.error.emit(f"クリップ失敗: {result.stderr}")
                    return
                self.clip_done.emit(job["output"])

            self.progress.emit(100, "全クリップ出力完了")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────
# ワーカースレッド: クリップ結合
# ─────────────────────────────────────────────

class MergeWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)  # 出力ファイルパス
    error = pyqtSignal(str)

    def __init__(self, video_path: str, segments: list, output_path: str, pre: float, post: float):
        """
        segments: [{"start": float, "end": float, ...}, ...]
        """
        super().__init__()
        self.video_path = video_path
        self.segments = segments
        self.output_path = output_path
        self.pre = pre
        self.post = post

    def run(self):
        try:
            import uuid

            self.progress.emit(10, "クリップを抽出中...")

            # 一時ディレクトリ
            tmp_dir = "/tmp" if platform.system() != "Windows" else tempfile.gettempdir()
            tmp_clips = []

            # 各セグメントを一時ファイルに出力
            total = len(self.segments)
            for i, seg in enumerate(self.segments):
                start = max(0, seg["start"] - self.pre)
                end = seg["end"] + self.post
                duration = end - start

                tmp_clip = os.path.join(tmp_dir, f"clip_{uuid.uuid4().hex}.ts")
                tmp_clips.append(tmp_clip)

                cmd = [
                    get_ffmpeg_path(), "-y",
                    "-ss", str(start),
                    "-i", self.video_path,
                    "-t", str(duration),
                    "-c", "copy",
                    "-bsf:v", "h264_mp4toannexb",
                    "-f", "mpegts",
                    tmp_clip
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                if result.returncode != 0:
                    raise RuntimeError(f"クリップ抽出失敗: {result.stderr}")

                pct = 10 + int((i + 1) / total * 50)
                self.progress.emit(pct, f"クリップ {i+1}/{total} を抽出中...")

            self.progress.emit(70, "クリップを結合中...")

            # concatで結合
            concat_input = "|".join(tmp_clips)
            cmd = [
                get_ffmpeg_path(), "-y",
                "-i", f"concat:{concat_input}",
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                self.output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode != 0:
                raise RuntimeError(f"結合失敗: {result.stderr}")

            # 一時ファイル削除
            for tmp_clip in tmp_clips:
                try:
                    os.unlink(tmp_clip)
                except:
                    pass

            self.progress.emit(100, "結合完了")
            self.finished.emit(self.output_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ─────────────────────────────────────────────
# ワーカースレッド: 文字起こし＋話者分離統合
# ─────────────────────────────────────────────

class CombinedTranscribeWorker(QThread):
    """
    TranscribeWorkerとDiarizationWorkerを並列実行し、
    結果をマージしてセグメントに話者情報を付与する
    """
    progress = pyqtSignal(int, str)
    segment_ready = pyqtSignal(dict)
    diarization_started = pyqtSignal()  # 話者分離開始を通知
    finished = pyqtSignal(list, list)  # (segments, speakers)
    error = pyqtSignal(str)

    def __init__(self, video_path: str, model_size: str, language: str = "ja",
                 enable_diarization: bool = False, hf_token: str = "",
                 min_speakers: int = None, max_speakers: int = None,
                 fast_diarization: bool = False):
        super().__init__()
        self.video_path = video_path
        self.model_size = model_size
        self.language = language
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.fast_diarization = fast_diarization
        self._cancelled = False

        # 結果保存用
        self._transcribe_segments = []
        self._diarization_segments = []
        self._transcribe_done = False
        self._diarization_done = False
        self._audio_path = None

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import torch
            import numpy as np
            import wave
            import uuid

            self.progress.emit(5, "音声を抽出中...")

            # 一時WAVファイルに音声抽出
            tmp_dir = "/tmp" if platform.system() != "Windows" else tempfile.gettempdir()
            self._audio_path = os.path.join(tmp_dir, f"clipfinder_{uuid.uuid4().hex}.wav")

            cmd = [
                get_ffmpeg_path(), "-y", "-i", self.video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                self._audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg エラー: {result.stderr}")

            if self._cancelled:
                self._cleanup()
                return

            # 音声データを読み込み
            with wave.open(self._audio_path, 'rb') as wf:
                audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0

            # 文字起こし実行
            self.progress.emit(20, "音声認識モデルを読み込み中...")
            transcribe_segments = self._run_transcribe(audio_float, torch)

            if self._cancelled:
                self._cleanup()
                return

            # 話者分離実行（有効な場合）
            diarization_segments = []
            speakers = []
            speaker_name_map = {}
            if self.enable_diarization and self.hf_token:
                self.diarization_started.emit()  # 話者分離開始を通知
                self.progress.emit(60, "話者分離を実行中...")
                diarization_segments = self._run_diarization(torch)
                if diarization_segments:
                    # 話者IDを抽出してランダムな名前にマッピング
                    import random
                    raw_speakers = list(set(s["speaker"] for s in diarization_segments))
                    raw_speakers.sort()
                    available_names = SPEAKER_NAMES.copy()
                    random.shuffle(available_names)
                    for i, spk in enumerate(raw_speakers):
                        if i < len(available_names):
                            speaker_name_map[spk] = available_names[i]
                        else:
                            speaker_name_map[spk] = f"話者{i+1}"
                    # diarization_segmentsの話者名を置換
                    for seg in diarization_segments:
                        seg["speaker"] = speaker_name_map.get(seg["speaker"], seg["speaker"])
                    speakers = [speaker_name_map[spk] for spk in raw_speakers]

            if self._cancelled:
                self._cleanup()
                return

            # 結果をマージ
            self.progress.emit(90, "結果を統合中...")
            merged_segments = self._merge_results(transcribe_segments, diarization_segments)

            self._cleanup()

            self.progress.emit(100, "完了")
            self.finished.emit(merged_segments, speakers)

        except ImportError as e:
            import traceback
            traceback.print_exc()
            self.error.emit(
                "必要なパッケージが見つかりません。\n"
                "pip install faster-whisper torch pyannote.audio を実行してください。"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self._cleanup()

    def _run_transcribe(self, audio_float, torch):
        """faster-whisperで文字起こし"""
        from faster_whisper import WhisperModel

        # GPU/CPU自動選択
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"

        actual_model = self.model_size
        if self.model_size == "auto":
            actual_model = "large-v3" if torch.cuda.is_available() else "medium"

        self.progress.emit(30, f"Whisper {actual_model} ({device}) で認識中...")

        model = WhisperModel(actual_model, device=device, compute_type=compute_type)

        segment_generator, info = model.transcribe(
            audio_float,
            language=self.language,
            word_timestamps=True,
            vad_filter=True,
        )

        segments = []
        for i, seg in enumerate(segment_generator):
            if self._cancelled:
                break
            segment_data = {
                "id": i,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "speaker": None,  # 話者分離で後から設定
            }
            segments.append(segment_data)
            self.segment_ready.emit(segment_data)
            # セグメント総数が不明なため、進捗メッセージのみ更新（進捗値は固定）
            self.progress.emit(45, f"認識中... {i+1} セグメント")

        return segments

    def _run_diarization(self, torch):
        """pyannote-audioで話者分離"""
        try:
            from pyannote.audio import Pipeline

            # HF_TOKEN環境変数を設定（pyannote-audio 3.1+で必要）
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_name = "3.0（高速）" if self.fast_diarization else "3.1（高精度）"
            self.progress.emit(62, f"話者分離: モデル {model_name} を準備中...")

            # キャッシュからパイプラインを取得（DiarizationWorkerのキャッシュを共有）
            pipeline = DiarizationWorker.get_pipeline(self.hf_token, device, self.fast_diarization)

            diarization_params = {}
            if self.min_speakers is not None:
                diarization_params["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                diarization_params["max_speakers"] = self.max_speakers

            self.progress.emit(68, "話者分離: 音声埋め込み計算中... (1/3)")
            self.progress.emit(75, "話者分離: クラスタリング中... (2/3)")

            diarization = pipeline(self._audio_path, **diarization_params)

            self.progress.emit(85, "話者分離: ラベル割り当て中... (3/3)")

            segments = []
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            return segments

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                self.progress.emit(70, "話者分離: 認証エラー（スキップ）")
            else:
                self.progress.emit(70, f"話者分離エラー: {error_msg[:50]}")
            return []

    def _merge_results(self, transcribe_segments: list, diarization_segments: list) -> list:
        """文字起こしセグメントに話者情報を付与"""
        if not diarization_segments:
            return transcribe_segments

        for seg in transcribe_segments:
            seg_mid = (seg["start"] + seg["end"]) / 2

            # セグメントの中央時間が含まれる話者区間を探す
            best_speaker = None
            best_overlap = 0

            for d_seg in diarization_segments:
                # オーバーラップを計算
                overlap_start = max(seg["start"], d_seg["start"])
                overlap_end = min(seg["end"], d_seg["end"])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d_seg["speaker"]

            seg["speaker"] = best_speaker

        return transcribe_segments

    def _cleanup(self):
        """一時ファイルを削除"""
        if self._audio_path and os.path.exists(self._audio_path):
            try:
                os.unlink(self._audio_path)
            except:
                pass


# ─────────────────────────────────────────────
# 話者カラー管理
# ─────────────────────────────────────────────

SPEAKER_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light Blue
]


def get_speaker_color(speaker_id: str, speakers: list) -> str:
    """話者IDに対応する色を返す"""
    if not speaker_id or not speakers:
        return "#6050b8"  # デフォルト色
    try:
        idx = speakers.index(speaker_id)
        return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
    except ValueError:
        return "#6050b8"


# ─────────────────────────────────────────────
# セグメントカードウィジェット
# ─────────────────────────────────────────────

class SegmentCard(QFrame):
    clip_requested = pyqtSignal(dict)  # segment data
    selection_changed = pyqtSignal()   # 選択状態変更

    def __init__(self, segment: dict, highlight_word: str = "", speakers: list = None, config: ConfigManager = None, diarization_pending: bool = False):
        super().__init__()
        self.segment = segment
        self._speakers = speakers or []
        self._config = config
        self._diarization_pending = diarization_pending
        self.setObjectName("SegmentCard")
        self._speaker_label = None
        self._setup_ui(highlight_word)

    def _setup_ui(self, highlight_word: str):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # チェックボックス（結合用選択）
        self._checkbox = QCheckBox()
        self._checkbox.setFixedWidth(24)
        self._checkbox.stateChanged.connect(lambda: self.selection_changed.emit())

        # 話者ラベル（話者分離が有効な場合のみ表示）
        speaker = self.segment.get("speaker")
        if self._diarization_pending:
            # 話者分離中のローディング表示
            speaker_label = QLabel("分離中...")
            speaker_label.setObjectName("SpeakerLabel")
            speaker_label.setFixedWidth(70)
            speaker_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            speaker_label.setStyleSheet("""
                QLabel {
                    background: #3a3a5c;
                    color: #888;
                    border-radius: 4px;
                    padding: 2px 6px;
                    font-size: 11px;
                    font-style: italic;
                }
            """)
            self._speaker_label = speaker_label
        elif speaker and self._speakers:
            color = get_speaker_color(speaker, self._speakers)
            label_text = self._config.get_speaker_label(speaker) if self._config else speaker
            speaker_label = QLabel(label_text)
            speaker_label.setObjectName("SpeakerLabel")
            speaker_label.setFixedWidth(70)
            speaker_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            speaker_label.setStyleSheet(f"""
                QLabel {{
                    background: {color};
                    color: #1a1a2e;
                    border-radius: 4px;
                    padding: 2px 6px;
                    font-size: 11px;
                    font-weight: 600;
                }}
            """)
        else:
            speaker_label = None

        # タイムスタンプ
        start = self.segment["start"]
        end = self.segment["end"]
        ts_label = QLabel(f"{self._fmt(start)} → {self._fmt(end)}")
        ts_label.setObjectName("TimestampLabel")
        ts_label.setFixedWidth(140)

        # テキスト（ハイライト）
        text = self.segment["text"]
        text_label = QLabel(self._highlight(text, highlight_word))
        text_label.setTextFormat(Qt.TextFormat.RichText)
        text_label.setWordWrap(True)
        text_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )

        # クリップボタン
        clip_btn = QPushButton("クリップ")
        clip_btn.setObjectName("ClipBtn")
        clip_btn.setFixedWidth(72)
        clip_btn.clicked.connect(lambda: self.clip_requested.emit(self.segment))

        layout.addWidget(self._checkbox)
        if speaker_label:
            layout.addWidget(speaker_label)
        layout.addWidget(ts_label)
        layout.addWidget(text_label, 1)
        layout.addWidget(clip_btn)

    def is_selected(self) -> bool:
        return self._checkbox.isChecked()

    def set_selected(self, selected: bool):
        self._checkbox.setChecked(selected)

    def _fmt(self, seconds: float) -> str:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:05.2f}"

    def _highlight(self, text: str, word: str) -> str:
        if not word:
            return text
        import re
        escaped = re.escape(word)
        highlighted = re.sub(
            escaped,
            f'<span style="background:#f0c040;color:#1a1a2e;border-radius:3px;padding:1px 3px;">{word}</span>',
            text,
            flags=re.IGNORECASE
        )
        return highlighted

    def mark_matched(self):
        self.setProperty("matched", True)
        self.style().unpolish(self)
        self.style().polish(self)


# ─────────────────────────────────────────────
# メインウィンドウ
# ─────────────────────────────────────────────

class ClipFinderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClipFinder — ゲーム動画セリフクリッパー")
        self.setMinimumSize(900, 660)
        self.resize(1100, 720)

        self._video_path = ""
        self._segments: list[dict] = []
        self._speakers: list[str] = []  # 検出された話者リスト
        self._speaker_filter: str = None  # 現在の話者フィルタ（Noneは全て表示）
        self._diarization_pending: bool = False  # 話者分離中フラグ
        self._transcribe_worker = None
        self._clip_worker = None
        self._merge_worker = None
        self._config = ConfigManager()

        self._apply_stylesheet()
        self._build_ui()
        self._setup_statusbar()
        self._load_settings()

    # ── スタイル ──────────────────────────────

    def _apply_stylesheet(self):
        self.setStyleSheet("""
        QMainWindow {
            background: #0f0f1a;
        }
        QWidget {
            background: #0f0f1a;
            color: #e0deff;
            font-family: 'Segoe UI', 'Noto Sans JP', sans-serif;
            font-size: 13px;
        }

        /* ヘッダー */
        #AppTitle {
            font-size: 22px;
            font-weight: 700;
            color: #c8b8ff;
            letter-spacing: 1px;
        }
        #AppSubtitle {
            font-size: 11px;
            color: #6658aa;
        }

        /* ドロップゾーン */
        #DropZone {
            border: 2px dashed #3d3060;
            border-radius: 10px;
            background: #15122a;
            color: #7060b8;
            font-size: 14px;
            padding: 20px;
        }
        #DropZone:hover {
            border-color: #8060e0;
            color: #b090ff;
            background: #1a1535;
        }

        /* グループボックス */
        QGroupBox {
            border: 1px solid #2a2050;
            border-radius: 8px;
            margin-top: 8px;
            padding: 6px;
            color: #9080d8;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }

        /* 入力 */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #1a1535;
            border: 1px solid #2d2460;
            border-radius: 6px;
            padding: 5px 10px;
            color: #e0deff;
            selection-background-color: #6040c0;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #8060e0;
            background: #1e1840;
        }

        /* プライマリボタン */
        #PrimaryBtn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #6040c0, stop:1 #8050e8);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 20px;
            font-weight: 700;
            font-size: 13px;
        }
        #PrimaryBtn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #7050d8, stop:1 #9060f8);
        }
        #PrimaryBtn:pressed {
            background: #5030a8;
        }
        #PrimaryBtn:disabled {
            background: #2a2050;
            color: #5048a0;
        }

        /* セカンダリボタン */
        #SecondaryBtn {
            background: #1e1840;
            color: #b0a0ff;
            border: 1px solid #3d3060;
            border-radius: 8px;
            padding: 8px 16px;
        }
        #SecondaryBtn:hover {
            background: #261e50;
            border-color: #6040c0;
        }

        /* クリップボタン */
        #ClipBtn {
            background: #1a3040;
            color: #60c8ff;
            border: 1px solid #204050;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 12px;
        }
        #ClipBtn:hover {
            background: #204860;
            border-color: #40a8e0;
        }

        /* プログレスバー */
        QProgressBar {
            background: #1a1535;
            border: 1px solid #2a2050;
            border-radius: 6px;
            text-align: center;
            color: #c8b8ff;
            height: 18px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #5030b0, stop:1 #8060e8);
            border-radius: 5px;
        }

        /* セグメントカード */
        #SegmentCard {
            background: #13102a;
            border: 1px solid #221e48;
            border-radius: 8px;
            margin: 2px 4px;
        }
        #SegmentCard:hover {
            background: #18143a;
            border-color: #4030a0;
        }
        #SegmentCard[matched="true"] {
            border-color: #8060e0;
            background: #1a1640;
        }

        /* タイムスタンプ */
        #TimestampLabel {
            color: #6050b8;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
        }

        /* スクロールバー */
        QScrollBar:vertical {
            background: #0f0f1a;
            width: 8px;
        }
        QScrollBar::handle:vertical {
            background: #3d3060;
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #6040c0;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }

        /* ステータスバー */
        QStatusBar {
            background: #0a0a14;
            color: #6050b8;
            border-top: 1px solid #1e1840;
            font-size: 11px;
        }

        /* タブ */
        QTabWidget::pane {
            border: 1px solid #2a2050;
            border-radius: 6px;
        }
        QTabBar::tab {
            background: #13102a;
            color: #6050a8;
            padding: 7px 16px;
            border: 1px solid #2a2050;
            border-bottom: none;
            border-radius: 6px 6px 0 0;
        }
        QTabBar::tab:selected {
            background: #1e1840;
            color: #c8b8ff;
            border-color: #4030a0;
        }

        /* リスト（クリップ履歴） */
        QListWidget {
            background: #13102a;
            border: 1px solid #2a2050;
            border-radius: 6px;
        }
        QListWidget::item {
            padding: 6px;
            border-bottom: 1px solid #1e1840;
        }
        QListWidget::item:selected {
            background: #2a2050;
        }

        /* ラベル系 */
        #SectionLabel {
            color: #7060b8;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        """)

    # ── UI構築 ──────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 8)
        root.setSpacing(10)

        # ヘッダー
        root.addLayout(self._build_header())

        # メインスプリッタ
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([360, 700])
        splitter.setHandleWidth(2)

        root.addWidget(splitter, 1)

    def _build_header(self):
        h = QHBoxLayout()
        title = QLabel("ClipFinder")
        title.setObjectName("AppTitle")
        sub = QLabel("Game Clip Cutter — Powered by Whisper")
        sub.setObjectName("AppSubtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignBottom)

        h.addWidget(title)
        h.addWidget(sub)
        h.addStretch()
        return h

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # ── 動画選択 ─────────────────────────
        video_group = QGroupBox("動画ファイル")
        vg_layout = QVBoxLayout(video_group)

        self._drop_label = QLabel("ここにファイルをドラッグ\nまたは下のボタンから選択")
        self._drop_label.setObjectName("DropZone")
        self._drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_label.setMinimumHeight(80)
        self._drop_label.setAcceptDrops(True)

        select_btn = QPushButton("ファイルを選択...")
        select_btn.setObjectName("SecondaryBtn")
        select_btn.clicked.connect(self._select_video)

        vg_layout.addWidget(self._drop_label)
        vg_layout.addWidget(select_btn)
        layout.addWidget(video_group)

        # ── 認識設定 ─────────────────────────
        settings_group = QGroupBox("認識設定")
        sg_layout = QVBoxLayout(settings_group)

        # モデル選択
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Whisperモデル"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["auto (GPU→large / CPU→medium)", "tiny", "base", "small", "medium", "large"])
        model_row.addWidget(self._model_combo, 1)
        sg_layout.addLayout(model_row)

        # 言語
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("言語"))
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(["ja (日本語)", "en (英語)", "auto"])
        lang_row.addWidget(self._lang_combo, 1)
        sg_layout.addLayout(lang_row)

        layout.addWidget(settings_group)

        # ── 話者分離設定 ─────────────────────────
        diarization_group = QGroupBox("話者分離")
        dg_layout = QVBoxLayout(diarization_group)

        # 話者分離ON/OFFトグル
        self._diarization_check = QCheckBox("話者分離を有効にする")
        self._diarization_check.stateChanged.connect(self._on_diarization_toggled)
        dg_layout.addWidget(self._diarization_check)

        # 高速モードチェックボックス
        self._diarization_fast_mode = QCheckBox("高速モード (v3.0)")
        self._diarization_fast_mode.setToolTip("精度は少し下がりますが、処理速度が向上します")
        dg_layout.addWidget(self._diarization_fast_mode)

        # 話者数ヒント
        speakers_row = QHBoxLayout()
        speakers_row.addWidget(QLabel("話者数"))
        self._min_speakers_spin = QSpinBox()
        self._min_speakers_spin.setRange(0, 20)
        self._min_speakers_spin.setValue(0)
        self._min_speakers_spin.setSpecialValueText("自動")
        self._min_speakers_spin.setToolTip("最小話者数（0=自動）")
        speakers_row.addWidget(self._min_speakers_spin)
        speakers_row.addWidget(QLabel("〜"))
        self._max_speakers_spin = QSpinBox()
        self._max_speakers_spin.setRange(0, 20)
        self._max_speakers_spin.setValue(2)  # デフォルト2
        self._max_speakers_spin.setSpecialValueText("自動")
        self._max_speakers_spin.setToolTip("最大話者数（0=自動）")
        speakers_row.addWidget(self._max_speakers_spin)
        dg_layout.addLayout(speakers_row)

        # 初期状態で話者分離設定を無効化
        self._min_speakers_spin.setEnabled(False)
        self._max_speakers_spin.setEnabled(False)
        self._diarization_fast_mode.setEnabled(False)

        layout.addWidget(diarization_group)

        # ── クリップ設定 ──────────────────────
        clip_group = QGroupBox("クリップ設定")
        cg_layout = QVBoxLayout(clip_group)

        pre_row = QHBoxLayout()
        pre_row.addWidget(QLabel("前に追加（秒）"))
        self._pre_spin = QDoubleSpinBox()
        self._pre_spin.setRange(0, 60)
        self._pre_spin.setValue(3.0)
        self._pre_spin.setSingleStep(0.5)
        pre_row.addWidget(self._pre_spin)
        cg_layout.addLayout(pre_row)

        post_row = QHBoxLayout()
        post_row.addWidget(QLabel("後に追加（秒）"))
        self._post_spin = QDoubleSpinBox()
        self._post_spin.setRange(0, 60)
        self._post_spin.setValue(3.0)
        self._post_spin.setSingleStep(0.5)
        post_row.addWidget(self._post_spin)
        cg_layout.addLayout(post_row)

        # 出力ディレクトリ
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("出力先"))
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("動画と同じフォルダ")
        out_btn = QPushButton("...")
        out_btn.setFixedWidth(30)
        out_btn.setObjectName("SecondaryBtn")
        out_btn.clicked.connect(self._select_output_dir)
        out_row.addWidget(self._out_dir_edit, 1)
        out_row.addWidget(out_btn)
        cg_layout.addLayout(out_row)

        layout.addWidget(clip_group)

        # ── 認識開始ボタン ─────────────────────
        self._transcribe_btn = QPushButton("音声認識を開始")
        self._transcribe_btn.setObjectName("PrimaryBtn")
        self._transcribe_btn.setMinimumHeight(40)
        self._transcribe_btn.clicked.connect(self._start_transcribe)
        layout.addWidget(self._transcribe_btn)

        # キャンセルボタン
        self._cancel_btn = QPushButton("中止")
        self._cancel_btn.setObjectName("SecondaryBtn")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_transcribe)
        layout.addWidget(self._cancel_btn)

        # プログレスバー
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setObjectName("SectionLabel")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._progress_label)

        layout.addStretch()
        return panel

    def _build_right_panel(self):
        tab = QTabWidget()

        # ── タブ1: 検索・クリップ ──────────────
        search_tab = QWidget()
        sl = QVBoxLayout(search_tab)
        sl.setSpacing(8)

        # 検索バー
        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("検索するセリフ・キーワードを入力...")
        self._search_edit.returnPressed.connect(self._do_search)
        search_btn = QPushButton("検索")
        search_btn.setObjectName("PrimaryBtn")
        search_btn.setFixedWidth(70)
        search_btn.clicked.connect(self._do_search)
        search_row.addWidget(self._search_edit, 1)
        search_row.addWidget(search_btn)
        sl.addLayout(search_row)

        # 検索結果ヘッダー
        result_header = QHBoxLayout()
        self._result_count_label = QLabel("認識結果: 0 件")
        self._result_count_label.setObjectName("SectionLabel")
        clip_all_btn = QPushButton("一致を全てクリップ")
        clip_all_btn.setObjectName("SecondaryBtn")
        clip_all_btn.clicked.connect(self._clip_all_matched)
        result_header.addWidget(self._result_count_label)
        result_header.addStretch()
        result_header.addWidget(clip_all_btn)
        sl.addLayout(result_header)

        # 話者フィルタ行（話者分離有効時のみ表示）
        self._speaker_filter_widget = QWidget()
        speaker_filter_layout = QHBoxLayout(self._speaker_filter_widget)
        speaker_filter_layout.setContentsMargins(0, 0, 0, 0)
        speaker_filter_layout.setSpacing(4)
        speaker_filter_label = QLabel("話者:")
        speaker_filter_label.setObjectName("SectionLabel")
        speaker_filter_layout.addWidget(speaker_filter_label)
        self._speaker_filter_buttons_layout = QHBoxLayout()
        self._speaker_filter_buttons_layout.setSpacing(4)
        speaker_filter_layout.addLayout(self._speaker_filter_buttons_layout)
        speaker_filter_layout.addStretch()
        # 「この話者のみクリップ」ボタン
        self._clip_speaker_btn = QPushButton("この話者のみクリップ")
        self._clip_speaker_btn.setObjectName("SecondaryBtn")
        self._clip_speaker_btn.clicked.connect(self._clip_filtered_speaker)
        self._clip_speaker_btn.setVisible(False)
        speaker_filter_layout.addWidget(self._clip_speaker_btn)
        self._speaker_filter_widget.setVisible(False)
        sl.addWidget(self._speaker_filter_widget)

        # 選択・結合コントロール
        select_row = QHBoxLayout()
        select_all_btn = QPushButton("全選択")
        select_all_btn.setObjectName("SecondaryBtn")
        select_all_btn.setFixedWidth(70)
        select_all_btn.clicked.connect(self._select_all_segments)
        deselect_btn = QPushButton("選択解除")
        deselect_btn.setObjectName("SecondaryBtn")
        deselect_btn.setFixedWidth(90)
        deselect_btn.clicked.connect(self._deselect_all_segments)
        self._selected_count_label = QLabel("選択: 0 件")
        self._selected_count_label.setObjectName("SectionLabel")
        merge_btn = QPushButton("選択を結合出力")
        merge_btn.setObjectName("PrimaryBtn")
        merge_btn.clicked.connect(self._merge_selected)
        select_row.addWidget(select_all_btn)
        select_row.addWidget(deselect_btn)
        select_row.addWidget(self._selected_count_label)
        select_row.addStretch()
        select_row.addWidget(merge_btn)
        sl.addLayout(select_row)

        # セグメントリスト（スクロール）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._segments_container = QWidget()
        self._segments_layout = QVBoxLayout(self._segments_container)
        self._segments_layout.setSpacing(2)
        self._segments_layout.addStretch()
        scroll.setWidget(self._segments_container)
        sl.addWidget(scroll, 1)

        tab.addTab(search_tab, "検索 & クリップ")

        # ── タブ2: 全文字起こし ────────────────
        full_tab = QWidget()
        fl = QVBoxLayout(full_tab)
        self._full_text = QTextEdit()
        self._full_text.setReadOnly(True)
        self._full_text.setPlaceholderText("認識完了後にここに全文が表示されます")
        fl.addWidget(self._full_text)
        copy_btn = QPushButton("全文をコピー")
        copy_btn.setObjectName("SecondaryBtn")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self._full_text.toPlainText()))
        fl.addWidget(copy_btn)
        tab.addTab(full_tab, "全文字起こし")

        # ── タブ3: クリップ履歴 ────────────────
        history_tab = QWidget()
        hl = QVBoxLayout(history_tab)
        self._clip_list = QListWidget()
        hl.addWidget(self._clip_list)
        open_folder_btn = QPushButton("出力フォルダを開く")
        open_folder_btn.setObjectName("SecondaryBtn")
        open_folder_btn.clicked.connect(self._open_output_folder)
        hl.addWidget(open_folder_btn)
        tab.addTab(history_tab, "クリップ履歴")

        return tab

    def _setup_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("動画ファイルを選択してください")

    def _load_settings(self):
        """設定を読み込んでUIに反映"""
        self._diarization_check.setChecked(self._config.diarization_enabled)
        self._on_diarization_toggled()

    def _on_diarization_toggled(self):
        """話者分離の有効/無効切り替え"""
        enabled = self._diarization_check.isChecked()
        self._min_speakers_spin.setEnabled(enabled)
        self._max_speakers_spin.setEnabled(enabled)
        self._diarization_fast_mode.setEnabled(enabled)
        self._config.diarization_enabled = enabled

    # ── イベントハンドラ ──────────────────────

    def _select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "動画を選択",
            "",
            "動画ファイル (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.ts);;全てのファイル (*)"
        )
        if path:
            self._set_video(path)

    def _set_video(self, path: str):
        self._video_path = path
        name = Path(path).name
        self._drop_label.setText(f"{name}")
        self._drop_label.setToolTip(path)
        self._statusbar.showMessage(f"読み込み: {name}")
        self._segments = []
        self._speakers = []
        self._speaker_filter = None
        self._clear_segments_ui()
        self._clear_speaker_filter_buttons()
        self._full_text.clear()

    def _select_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "出力フォルダを選択")
        if d:
            self._out_dir_edit.setText(d)

    def _start_transcribe(self):
        if not self._video_path:
            QMessageBox.warning(self, "エラー", "動画ファイルを選択してください")
            return

        model_text = self._model_combo.currentText()
        model_key = model_text.split()[0] if " " in model_text else model_text

        lang_text = self._lang_combo.currentText()
        lang = lang_text.split()[0]
        if lang == "auto":
            lang = None

        self._segments = []
        self._speakers = []
        self._speaker_filter = None
        self._clear_segments_ui()
        self._clear_speaker_filter_buttons()
        self._full_text.clear()
        self._result_count_label.setText("認識結果: 0 件")

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._progress_label.setText("準備中...")
        self._transcribe_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)

        # 話者分離オプション
        enable_diarization = self._diarization_check.isChecked()
        min_speakers = self._min_speakers_spin.value() if self._min_speakers_spin.value() > 0 else None
        max_speakers = self._max_speakers_spin.value() if self._max_speakers_spin.value() > 0 else None
        fast_diarization = self._diarization_fast_mode.isChecked()

        self._transcribe_worker = CombinedTranscribeWorker(
            self._video_path, model_key, lang or "ja",
            enable_diarization=enable_diarization,
            hf_token=HF_TOKEN,  # 固定トークン使用
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            fast_diarization=fast_diarization
        )
        self._diarization_pending = enable_diarization  # 話者分離待ちフラグ
        self._transcribe_worker.progress.connect(self._on_transcribe_progress)
        self._transcribe_worker.segment_ready.connect(self._on_segment_ready)
        self._transcribe_worker.diarization_started.connect(self._on_diarization_started)
        self._transcribe_worker.finished.connect(self._on_transcribe_done_with_speakers)
        self._transcribe_worker.error.connect(self._on_transcribe_error)
        self._transcribe_worker.start()

    def _cancel_transcribe(self):
        if self._transcribe_worker:
            self._transcribe_worker.cancel()
        self._diarization_pending = False
        self._on_transcribe_reset()
        self._statusbar.showMessage("認識をキャンセルしました")

    def _on_transcribe_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_segment_ready(self, seg: dict):
        self._segments.append(seg)
        self._add_segment_card(seg, diarization_pending=False)  # まだ話者分離前
        self._result_count_label.setText(f"認識結果: {len(self._segments)} 件")

    def _on_diarization_started(self):
        """話者分離が開始されたら、既存のセグメントカードを更新"""
        self._diarization_pending = True
        # 既存のセグメントカードを再構築して「分離中...」表示に
        self._clear_segments_ui()
        for seg in self._segments:
            self._add_segment_card(seg, diarization_pending=True)

    def _on_transcribe_done(self, segments: list):
        self._segments = segments
        full = "\n".join(
            f"[{self._fmt_time(s['start'])}] {s['text']}" for s in segments
        )
        self._full_text.setPlainText(full)
        self._on_transcribe_reset()
        self._statusbar.showMessage(f"認識完了: {len(segments)} セグメント")

    def _on_transcribe_done_with_speakers(self, segments: list, speakers: list):
        """話者情報付きの認識完了ハンドラ"""
        self._diarization_pending = False  # 話者分離完了
        self._segments = segments
        self._speakers = speakers

        # 全文表示（話者情報付き）
        lines = []
        for s in segments:
            speaker_str = f"[{s.get('speaker', '?')}] " if s.get('speaker') else ""
            lines.append(f"[{self._fmt_time(s['start'])}] {speaker_str}{s['text']}")
        self._full_text.setPlainText("\n".join(lines))

        # セグメントカードを再構築
        self._clear_segments_ui()
        for seg in segments:
            self._add_segment_card(seg)

        # 話者フィルタボタンを構築
        if speakers:
            self._build_speaker_filter_buttons()
            self._speaker_filter_widget.setVisible(True)
        else:
            self._speaker_filter_widget.setVisible(False)

        self._result_count_label.setText(f"認識結果: {len(segments)} 件")
        self._on_transcribe_reset()

        status_msg = f"認識完了: {len(segments)} セグメント"
        if speakers:
            status_msg += f", {len(speakers)} 話者検出"
        self._statusbar.showMessage(status_msg)

    def _on_transcribe_error(self, msg: str):
        QMessageBox.critical(self, "認識エラー", msg)
        self._on_transcribe_reset()

    def _on_transcribe_reset(self):
        self._transcribe_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        QTimer.singleShot(2000, lambda: self._progress_bar.setVisible(False))
        QTimer.singleShot(2000, lambda: self._progress_label.setText(""))

    # ── セグメントUI ──────────────────────────

    def _clear_segments_ui(self):
        while self._segments_layout.count() > 1:  # stretchを残す
            item = self._segments_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_segment_card(self, seg: dict, highlight: str = "", diarization_pending: bool = False):
        # 話者フィルタが有効な場合、該当しないセグメントはスキップ
        if self._speaker_filter and seg.get("speaker") != self._speaker_filter:
            return

        card = SegmentCard(seg, highlight, speakers=self._speakers, config=self._config, diarization_pending=diarization_pending)
        card.clip_requested.connect(self._on_clip_requested)
        card.selection_changed.connect(self._update_selection_count)
        # stretch前に挿入
        pos = self._segments_layout.count() - 1
        self._segments_layout.insertWidget(pos, card)

    def _clear_speaker_filter_buttons(self):
        """話者フィルタボタンをクリア"""
        while self._speaker_filter_buttons_layout.count() > 0:
            item = self._speaker_filter_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._speaker_filter_widget.setVisible(False)
        self._clip_speaker_btn.setVisible(False)

    def _build_speaker_filter_buttons(self):
        """話者フィルタボタンを構築"""
        self._clear_speaker_filter_buttons()

        # 「全て」ボタン
        all_btn = QPushButton("全て")
        all_btn.setObjectName("SpeakerFilterBtn")
        all_btn.setCheckable(True)
        all_btn.setChecked(True)
        all_btn.setStyleSheet(self._get_speaker_btn_style(None, checked=True))
        all_btn.clicked.connect(lambda: self._on_speaker_filter_clicked(None))
        self._speaker_filter_buttons_layout.addWidget(all_btn)

        # 各話者ボタン
        for speaker in self._speakers:
            color = get_speaker_color(speaker, self._speakers)
            label = self._config.get_speaker_label(speaker)
            btn = QPushButton(label)
            btn.setObjectName("SpeakerFilterBtn")
            btn.setCheckable(True)
            btn.setStyleSheet(self._get_speaker_btn_style(color, checked=False))
            btn.clicked.connect(lambda checked, s=speaker: self._on_speaker_filter_clicked(s))
            self._speaker_filter_buttons_layout.addWidget(btn)

    def _get_speaker_btn_style(self, color: str, checked: bool) -> str:
        """話者フィルタボタンのスタイルを生成"""
        if color:
            if checked:
                return f"""
                    QPushButton {{
                        background: {color};
                        color: #1a1a2e;
                        border: 2px solid {color};
                        border-radius: 4px;
                        padding: 4px 8px;
                        font-weight: 600;
                    }}
                """
            else:
                return f"""
                    QPushButton {{
                        background: transparent;
                        color: {color};
                        border: 1px solid {color};
                        border-radius: 4px;
                        padding: 4px 8px;
                    }}
                    QPushButton:hover {{
                        background: {color}40;
                    }}
                """
        else:
            # 「全て」ボタン
            if checked:
                return """
                    QPushButton {
                        background: #6040c0;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 4px 8px;
                        font-weight: 600;
                    }
                """
            else:
                return """
                    QPushButton {
                        background: transparent;
                        color: #b0a0ff;
                        border: 1px solid #3d3060;
                        border-radius: 4px;
                        padding: 4px 8px;
                    }
                    QPushButton:hover {
                        background: #261e50;
                    }
                """

    def _on_speaker_filter_clicked(self, speaker: str):
        """話者フィルタボタンがクリックされた"""
        self._speaker_filter = speaker

        # ボタンの見た目を更新
        for i in range(self._speaker_filter_buttons_layout.count()):
            btn = self._speaker_filter_buttons_layout.itemAt(i).widget()
            if btn:
                if i == 0:  # 「全て」ボタン
                    checked = speaker is None
                    btn.setStyleSheet(self._get_speaker_btn_style(None, checked))
                else:
                    s = self._speakers[i - 1]
                    color = get_speaker_color(s, self._speakers)
                    checked = s == speaker
                    btn.setStyleSheet(self._get_speaker_btn_style(color, checked))

        # 「この話者のみクリップ」ボタンの表示切り替え
        self._clip_speaker_btn.setVisible(speaker is not None)

        # セグメントカードを再構築
        self._rebuild_segment_cards()

    def _rebuild_segment_cards(self):
        """現在のフィルタ設定でセグメントカードを再構築"""
        query = self._search_edit.text().strip()
        self._clear_segments_ui()

        matched = 0
        displayed = 0
        for seg in self._segments:
            # 話者フィルタ
            if self._speaker_filter and seg.get("speaker") != self._speaker_filter:
                continue

            displayed += 1
            highlight = query if query and query.lower() in seg["text"].lower() else ""
            card = SegmentCard(seg, highlight, speakers=self._speakers, config=self._config)
            if highlight:
                card.mark_matched()
                matched += 1
            card.clip_requested.connect(self._on_clip_requested)
            card.selection_changed.connect(self._update_selection_count)
            pos = self._segments_layout.count() - 1
            self._segments_layout.insertWidget(pos, card)

        # 結果ラベル更新
        if self._speaker_filter:
            label_text = f"表示: {displayed} 件 / 全 {len(self._segments)} 件"
        else:
            label_text = f"認識結果: {len(self._segments)} 件"
        if matched > 0:
            label_text += f" / 一致: {matched} 件"
        self._result_count_label.setText(label_text)

    def _clip_filtered_speaker(self):
        """現在フィルタ中の話者のセグメントを全てクリップ"""
        if not self._speaker_filter or not self._video_path:
            return

        speaker_segments = [s for s in self._segments if s.get("speaker") == self._speaker_filter]
        if not speaker_segments:
            QMessageBox.information(self, "情報", "該当するセグメントがありません")
            return

        self._export_clips(speaker_segments)

    def _do_search(self):
        query = self._search_edit.text().strip()
        if not query:
            # 検索クエリがない場合は_rebuild_segment_cardsを使用
            self._rebuild_segment_cards()
            return

        # 既存カードを再構築（ハイライト反映 + 話者フィルタ）
        self._clear_segments_ui()

        matched = 0
        displayed = 0
        for seg in self._segments:
            # 話者フィルタ
            if self._speaker_filter and seg.get("speaker") != self._speaker_filter:
                continue

            displayed += 1
            is_match = query.lower() in seg["text"].lower()
            highlight = query if is_match else ""
            card = SegmentCard(seg, highlight, speakers=self._speakers, config=self._config)
            if is_match:
                card.mark_matched()
                matched += 1
            card.clip_requested.connect(self._on_clip_requested)
            card.selection_changed.connect(self._update_selection_count)
            pos = self._segments_layout.count() - 1
            self._segments_layout.insertWidget(pos, card)

        # 結果ラベル更新
        if self._speaker_filter:
            label_text = f"表示: {displayed} 件 / 全 {len(self._segments)} 件 / 一致: {matched} 件"
        else:
            label_text = f"認識結果: {len(self._segments)} 件 / 一致: {matched} 件"
        self._result_count_label.setText(label_text)
        self._statusbar.showMessage(f'「{query}」で {matched} 件ヒット')

    def _on_clip_requested(self, seg: dict):
        if not self._video_path:
            return
        self._export_clips([seg])

    def _clip_all_matched(self):
        query = self._search_edit.text().strip()
        if not query or not self._segments:
            return

        # 話者フィルタも考慮
        matched = []
        for s in self._segments:
            if self._speaker_filter and s.get("speaker") != self._speaker_filter:
                continue
            if query.lower() in s["text"].lower():
                matched.append(s)

        if not matched:
            QMessageBox.information(self, "結果なし", "一致するセグメントがありません")
            return
        self._export_clips(matched)

    def _get_segment_cards(self) -> list:
        """全てのSegmentCardを取得"""
        cards = []
        for i in range(self._segments_layout.count()):
            widget = self._segments_layout.itemAt(i).widget()
            if isinstance(widget, SegmentCard):
                cards.append(widget)
        return cards

    def _select_all_segments(self):
        for card in self._get_segment_cards():
            card.set_selected(True)
        self._update_selection_count()

    def _deselect_all_segments(self):
        for card in self._get_segment_cards():
            card.set_selected(False)
        self._update_selection_count()

    def _update_selection_count(self):
        selected = sum(1 for card in self._get_segment_cards() if card.is_selected())
        self._selected_count_label.setText(f"選択: {selected} 件")

    def _merge_selected(self):
        if not self._video_path:
            QMessageBox.warning(self, "エラー", "動画ファイルを選択してください")
            return

        selected_cards = [card for card in self._get_segment_cards() if card.is_selected()]
        if not selected_cards:
            QMessageBox.warning(self, "エラー", "結合するセグメントを選択してください")
            return

        # 時間順にソート
        selected_segments = sorted([card.segment for card in selected_cards], key=lambda s: s["start"])

        # 出力パス決定
        out_dir = self._out_dir_edit.text().strip()
        if not out_dir:
            out_dir = str(Path(self._video_path).parent)

        ext = Path(self._video_path).suffix
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(Path(out_dir) / f"merged_{ts}{ext}")

        pre = self._pre_spin.value()
        post = self._post_spin.value()

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        self._merge_worker = MergeWorker(self._video_path, selected_segments, out_path, pre, post)
        self._merge_worker.progress.connect(lambda p, m: (
            self._progress_bar.setValue(p),
            self._progress_label.setText(m)
        ))
        self._merge_worker.finished.connect(self._on_merge_done)
        self._merge_worker.error.connect(lambda e: QMessageBox.critical(self, "結合エラー", e))
        self._merge_worker.start()

    def _on_merge_done(self, path: str):
        item = QListWidgetItem(f"[結合] {Path(path).name}")
        item.setToolTip(path)
        self._clip_list.insertItem(0, item)
        self._statusbar.showMessage(f"結合完了: {Path(path).name}")
        QTimer.singleShot(3000, lambda: self._progress_bar.setVisible(False))

    def _export_clips(self, segments: list):
        pre = self._pre_spin.value()
        post = self._post_spin.value()

        out_dir = self._out_dir_edit.text().strip()
        if not out_dir:
            out_dir = str(Path(self._video_path).parent)

        ext = Path(self._video_path).suffix
        jobs = []
        for seg in segments:
            start = max(0, seg["start"] - pre)
            end = seg["end"] + post
            ts = datetime.now().strftime("%H%M%S")
            safe_text = "".join(c for c in seg["text"][:20] if c.isalnum() or c in " _-")
            fname = f"clip_{ts}_{safe_text.strip()}{ext}"
            out_path = str(Path(out_dir) / fname)
            jobs.append({
                "video": self._video_path,
                "start": start,
                "end": end,
                "output": out_path
            })

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        self._clip_worker = ClipWorker(jobs)
        self._clip_worker.progress.connect(lambda p, m: (
            self._progress_bar.setValue(p),
            self._progress_label.setText(m)
        ))
        self._clip_worker.clip_done.connect(self._on_clip_done)
        self._clip_worker.finished.connect(self._on_clip_finished)
        self._clip_worker.error.connect(lambda e: QMessageBox.critical(self, "クリップエラー", e))
        self._clip_worker.start()

    def _on_clip_done(self, path: str):
        item = QListWidgetItem(Path(path).name)
        item.setToolTip(path)
        self._clip_list.insertItem(0, item)
        self._last_clip_path = path

    def _on_clip_finished(self):
        self._statusbar.showMessage("クリップ出力完了")
        QTimer.singleShot(3000, lambda: self._progress_bar.setVisible(False))

    def _open_output_folder(self):
        out_dir = self._out_dir_edit.text().strip()
        if not out_dir and self._video_path:
            out_dir = str(Path(self._video_path).parent)
        if out_dir and os.path.isdir(out_dir):
            system = platform.system()
            if system == "Windows":
                os.startfile(out_dir)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", out_dir])
            else:  # Linux
                subprocess.run(["xdg-open", out_dir])
        else:
            QMessageBox.information(self, "情報", "出力フォルダが見つかりません")

    # ── ユーティリティ ────────────────────────

    def _fmt_time(self, seconds: float) -> str:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:05.2f}"

    # ── ドラッグ&ドロップ ──────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._set_video(path)


# ─────────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ClipFinder")

    # Windows向けDPI対応
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)

    window = ClipFinderApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
