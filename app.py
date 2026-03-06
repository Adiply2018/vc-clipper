"""
ClipFinder - ゲーム動画セリフクリッパー
音声認識でセリフを検索し、前後をクリップするデスクトップアプリ
"""

import sys
import os
import json
import threading
import subprocess
import tempfile
import platform
from pathlib import Path
from datetime import datetime


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
            import whisper
            import torch

            self.progress.emit(5, "モデルを読み込み中...")

            # GPU自動選択
            device = "cuda" if torch.cuda.is_available() else "cpu"
            actual_model = self.model_size
            if self.model_size == "auto":
                actual_model = "large" if torch.cuda.is_available() else "medium"

            self.progress.emit(15, f"Whisper {actual_model} ({device}) で認識開始...")

            model = whisper.load_model(actual_model, device=device)

            if self._cancelled:
                return

            self.progress.emit(30, "音声を抽出中...")

            # 一時WAVファイルに音声抽出
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                get_ffmpeg_path(), "-y", "-i", self.video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                tmp_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg エラー: {result.stderr}")

            if self._cancelled:
                os.unlink(tmp_path)
                return

            self.progress.emit(50, "音声認識中（時間がかかります）...")

            result = model.transcribe(
                tmp_path,
                language=self.language,
                word_timestamps=True,
                verbose=False
            )

            os.unlink(tmp_path)

            if self._cancelled:
                return

            segments = []
            total = len(result["segments"])
            for i, seg in enumerate(result["segments"]):
                if self._cancelled:
                    break
                segment_data = {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                segments.append(segment_data)
                self.segment_ready.emit(segment_data)

                pct = 50 + int((i / max(total, 1)) * 45)
                self.progress.emit(pct, f"認識中... ({i+1}/{total})")

            self.progress.emit(100, "完了")
            self.finished.emit(segments)

        except ImportError:
            self.error.emit(
                "whisper が見つかりません。\n"
                "pip install openai-whisper torch を実行してください。"
            )
        except Exception as e:
            self.error.emit(str(e))


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
                    int(i / total * 100),
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
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.error.emit(f"クリップ失敗: {result.stderr}")
                    return
                self.clip_done.emit(job["output"])

            self.progress.emit(100, "全クリップ出力完了")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────
# セグメントカードウィジェット
# ─────────────────────────────────────────────

class SegmentCard(QFrame):
    clip_requested = pyqtSignal(dict)  # segment data

    def __init__(self, segment: dict, highlight_word: str = ""):
        super().__init__()
        self.segment = segment
        self.setObjectName("SegmentCard")
        self._setup_ui(highlight_word)

    def _setup_ui(self, highlight_word: str):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

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

        layout.addWidget(ts_label)
        layout.addWidget(text_label, 1)
        layout.addWidget(clip_btn)

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
        self._transcribe_worker = None
        self._clip_worker = None

        self._apply_stylesheet()
        self._build_ui()
        self._setup_statusbar()

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
        self._clear_segments_ui()
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
        self._clear_segments_ui()
        self._full_text.clear()
        self._result_count_label.setText("認識結果: 0 件")

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._progress_label.setText("準備中...")
        self._transcribe_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)

        self._transcribe_worker = TranscribeWorker(
            self._video_path, model_key, lang or "ja"
        )
        self._transcribe_worker.progress.connect(self._on_transcribe_progress)
        self._transcribe_worker.segment_ready.connect(self._on_segment_ready)
        self._transcribe_worker.finished.connect(self._on_transcribe_done)
        self._transcribe_worker.error.connect(self._on_transcribe_error)
        self._transcribe_worker.start()

    def _cancel_transcribe(self):
        if self._transcribe_worker:
            self._transcribe_worker.cancel()
        self._on_transcribe_reset()
        self._statusbar.showMessage("認識をキャンセルしました")

    def _on_transcribe_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_segment_ready(self, seg: dict):
        self._segments.append(seg)
        self._add_segment_card(seg)
        self._result_count_label.setText(f"認識結果: {len(self._segments)} 件")

    def _on_transcribe_done(self, segments: list):
        self._segments = segments
        full = "\n".join(
            f"[{self._fmt_time(s['start'])}] {s['text']}" for s in segments
        )
        self._full_text.setPlainText(full)
        self._on_transcribe_reset()
        self._statusbar.showMessage(f"認識完了: {len(segments)} セグメント")

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

    def _add_segment_card(self, seg: dict, highlight: str = ""):
        card = SegmentCard(seg, highlight)
        card.clip_requested.connect(self._on_clip_requested)
        # stretch前に挿入
        pos = self._segments_layout.count() - 1
        self._segments_layout.insertWidget(pos, card)

    def _do_search(self):
        query = self._search_edit.text().strip()
        if not query:
            return

        # 既存カードを再構築（ハイライト反映）
        self._clear_segments_ui()

        matched = 0
        for seg in self._segments:
            card = SegmentCard(seg, query if query.lower() in seg["text"].lower() else "")
            if query.lower() in seg["text"].lower():
                card.mark_matched()
                matched += 1
            card.clip_requested.connect(self._on_clip_requested)
            pos = self._segments_layout.count() - 1
            self._segments_layout.insertWidget(pos, card)

        self._result_count_label.setText(
            f"認識結果: {len(self._segments)} 件 / 一致: {matched} 件"
        )
        self._statusbar.showMessage(f'「{query}」で {matched} 件ヒット')

    def _on_clip_requested(self, seg: dict):
        if not self._video_path:
            return
        self._export_clips([seg])

    def _clip_all_matched(self):
        query = self._search_edit.text().strip()
        if not query or not self._segments:
            return
        matched = [s for s in self._segments if query.lower() in s["text"].lower()]
        if not matched:
            QMessageBox.information(self, "結果なし", "一致するセグメントがありません")
            return
        self._export_clips(matched)

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
