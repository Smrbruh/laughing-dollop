#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip install yt-dlp faster-whisper PyQt6

"""
YouTube Transcriber
-------------------
Downloads audio from YouTube and transcribes it using faster-whisper.
Outputs a .txt full transcript and .srt subtitle file with timestamps.

Dependencies:
    pip install yt-dlp faster-whisper PyQt6
    ffmpeg must be installed and on PATH (instructions shown in GUI if missing)
"""

# ──────────────────────────────────────────────
#  SECTION 1: IMPORTS
# ──────────────────────────────────────────────
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import (
    QObject, QSettings, QThread, Qt, pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QStatusBar, QTextEdit, QVBoxLayout,
    QWidget,
)

# ──────────────────────────────────────────────
#  SECTION 2: CONSTANTS & CONFIG
# ──────────────────────────────────────────────

APP_NAME = "YouTube Transcriber"
APP_VERSION = "1.0.0"
CONFIG_FILE = Path.home() / ".yt_transcriber_config.json"

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Russian": "ru",
    "Spanish": "es",
    "German": "de",
    "French": "fr",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it",
    "Turkish": "tr",
    "Polish": "pl",
    "Dutch": "nl",
}

MODELS = ["tiny", "base", "small", "medium", "large-v2"]
DEFAULT_MODEL = "small"
DEFAULT_OUTPUT = str(Path.home() / "Transcriptions")

# ──────────────────────────────────────────────
#  SECTION 3: HELPERS
# ──────────────────────────────────────────────

def load_config() -> dict:
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_config(data: dict) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace(" ", "_")
    return name[:120] or "transcript"


def seconds_to_srt_time(s: float) -> str:
    ms = int((s % 1) * 1000)
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seconds_to_srt_time(seg.start)
        end = seconds_to_srt_time(seg.end)
        lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(lines)


def segments_to_txt(segments) -> str:
    return " ".join(seg.text.strip() for seg in segments)


# ──────────────────────────────────────────────
#  SECTION 4: WORKER THREAD
# ──────────────────────────────────────────────

class TranscribeWorker(QObject):
    """Runs download + transcription in a background thread."""

    status_changed = pyqtSignal(str)      # human-readable step description
    progress_changed = pyqtSignal(int)    # 0-100
    finished = pyqtSignal(str, str, str)  # txt_path, srt_path, title
    error = pyqtSignal(str)

    def __init__(
        self,
        url: str,
        language: Optional[str],
        model_size: str,
        output_dir: str,
    ):
        super().__init__()
        self.url = url
        self.language = language
        self.model_size = model_size
        self.output_dir = output_dir
        self._cancelled = False
        self._process: Optional[subprocess.Popen] = None

    def cancel(self):
        self._cancelled = True
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass

    @pyqtSlot()
    def run(self):
        try:
            self._run()
        except Exception as exc:
            if not self._cancelled:
                self.error.emit(str(exc))

    def _run(self):
        import yt_dlp

        if self._cancelled:
            return

        # ── Step 1: resolve video metadata ──────────────────────────
        self.status_changed.emit("🔍  Fetching video info…")
        self.progress_changed.emit(5)

        ydl_opts_info = {"quiet": True, "skip_download": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                info = ydl.extract_info(self.url, download=False)
        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if "age" in msg.lower():
                raise RuntimeError("Video is age-restricted. Sign in via cookies.")
            if "private" in msg.lower():
                raise RuntimeError("Video is private or unavailable.")
            raise RuntimeError(f"Could not fetch video info:\n{msg}")

        if self._cancelled:
            return

        title = info.get("title", "video")
        safe_title = sanitize_filename(title)

        # ── Step 2: download audio ───────────────────────────────────
        self.status_changed.emit("⬇  Downloading audio…")
        self.progress_changed.emit(15)

        tmpdir = tempfile.mkdtemp(prefix="yt_transcriber_")
        audio_path = os.path.join(tmpdir, "audio.%(ext)s")

        ydl_opts_dl = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }
            ],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
                ydl.download([self.url])
        except yt_dlp.utils.DownloadError as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(f"Download failed:\n{e}")

        if self._cancelled:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return

        # find the resulting file
        audio_file = None
        for f in Path(tmpdir).iterdir():
            if f.suffix in {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".webm"}:
                audio_file = str(f)
                break

        if not audio_file:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError("Audio file not found after download. Is ffmpeg installed?")

        self.progress_changed.emit(40)

        # ── Step 3: load model ───────────────────────────────────────
        self.status_changed.emit(f"🤖  Loading {self.model_size} model…")

        from faster_whisper import WhisperModel

        # Use int8 for CPU (fastest), float16 for CUDA
        try:
            device = "cuda"
            compute = "float16"
            model = WhisperModel(self.model_size, device=device, compute_type=compute)
        except Exception:
            device = "cpu"
            compute = "int8"
            model = WhisperModel(self.model_size, device=device, compute_type=compute)

        if self._cancelled:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return

        self.progress_changed.emit(55)

        # ── Step 4: transcribe ───────────────────────────────────────
        lang_label = self.language or "auto"
        self.status_changed.emit(f"✍  Transcribing ({lang_label})…")

        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
        }
        if self.language:
            transcribe_kwargs["language"] = self.language

        segments_gen, info_result = model.transcribe(audio_file, **transcribe_kwargs)

        # Materialise generator (also allows progress tracking)
        segments = []
        duration = info_result.duration or 1.0
        for seg in segments_gen:
            if self._cancelled:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return
            segments.append(seg)
            pct = 55 + int((seg.end / duration) * 35)
            self.progress_changed.emit(min(pct, 90))

        shutil.rmtree(tmpdir, ignore_errors=True)

        if self._cancelled:
            return

        # ── Step 5: write output files ───────────────────────────────
        self.status_changed.emit("💾  Saving files…")
        self.progress_changed.emit(92)

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        base = out / safe_title
        txt_path = str(base.with_suffix(".txt"))
        srt_path = str(base.with_suffix(".srt"))

        Path(txt_path).write_text(segments_to_txt(segments), encoding="utf-8")
        Path(srt_path).write_text(segments_to_srt(segments), encoding="utf-8")

        self.progress_changed.emit(100)
        self.status_changed.emit("✅  Done!")
        self.finished.emit(txt_path, srt_path, title)


# ──────────────────────────────────────────────
#  SECTION 5: FFMPEG HELP DIALOG
# ──────────────────────────────────────────────

class FfmpegDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ffmpeg not found")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        lbl = QLabel(
            "<b>ffmpeg</b> is required but was not found on your PATH.<br><br>"
            "<b>Install instructions:</b>"
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setPlainText(
            "Windows:\n"
            "  winget install ffmpeg\n"
            "  — OR — download from https://ffmpeg.org/download.html\n"
            "  and add the bin/ folder to your PATH.\n\n"
            "macOS:\n"
            "  brew install ffmpeg\n\n"
            "Linux (Debian/Ubuntu):\n"
            "  sudo apt install ffmpeg\n\n"
            "Linux (Fedora):\n"
            "  sudo dnf install ffmpeg\n\n"
            "After installing, restart this app."
        )
        layout.addWidget(instructions)

        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)


# ──────────────────────────────────────────────
#  SECTION 6: MAIN WINDOW
# ──────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setMinimumWidth(560)
        self.setMinimumHeight(480)

        self._cfg = load_config()
        self._worker: Optional[TranscribeWorker] = None
        self._thread: Optional[QThread] = None
        self._result_txt: str = ""
        self._result_srt: str = ""

        self._build_ui()
        self._apply_theme()
        self._load_settings()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(24, 20, 24, 20)
        main.setSpacing(14)

        # Title
        title = QLabel(f"🎬  {APP_NAME}")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        main.addWidget(title)

        # URL row
        url_lbl = QLabel("YouTube URL")
        url_lbl.setFont(QFont("", 9, QFont.Weight.Medium))
        main.addWidget(url_lbl)

        url_row = QHBoxLayout()
        url_row.setSpacing(8)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=…")
        self.url_input.setMinimumHeight(36)
        url_row.addWidget(self.url_input)

        paste_btn = QPushButton("Paste")
        paste_btn.setFixedWidth(64)
        paste_btn.setMinimumHeight(36)
        paste_btn.clicked.connect(self._paste_url)
        url_row.addWidget(paste_btn)
        main.addLayout(url_row)

        # Options row
        opts_row = QHBoxLayout()
        opts_row.setSpacing(16)

        lang_col = QVBoxLayout()
        lang_col.setSpacing(4)
        lang_lbl = QLabel("Language")
        lang_lbl.setFont(QFont("", 9, QFont.Weight.Medium))
        lang_col.addWidget(lang_lbl)
        self.lang_box = QComboBox()
        self.lang_box.addItems(list(LANGUAGES.keys()))
        self.lang_box.setMinimumHeight(34)
        lang_col.addWidget(self.lang_box)
        opts_row.addLayout(lang_col)

        model_col = QVBoxLayout()
        model_col.setSpacing(4)
        model_lbl = QLabel("Model size")
        model_lbl.setFont(QFont("", 9, QFont.Weight.Medium))
        model_col.addWidget(model_lbl)
        self.model_box = QComboBox()
        self.model_box.addItems(MODELS)
        self.model_box.setCurrentText(DEFAULT_MODEL)
        self.model_box.setMinimumHeight(34)
        model_col.addWidget(self.model_box)
        opts_row.addLayout(model_col)
        main.addLayout(opts_row)

        # Output folder row
        out_lbl = QLabel("Output folder")
        out_lbl.setFont(QFont("", 9, QFont.Weight.Medium))
        main.addWidget(out_lbl)

        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self.out_input = QLineEdit()
        self.out_input.setMinimumHeight(36)
        self.out_input.setPlaceholderText(DEFAULT_OUTPUT)
        out_row.addWidget(self.out_input)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(72)
        browse_btn.setMinimumHeight(36)
        browse_btn.clicked.connect(self._browse_output)
        out_row.addWidget(browse_btn)
        main.addLayout(out_row)

        # Spacer
        main.addSpacing(4)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.transcribe_btn = QPushButton("▶  Transcribe")
        self.transcribe_btn.setMinimumHeight(42)
        self.transcribe_btn.setFont(QFont("", 11, QFont.Weight.Bold))
        self.transcribe_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.transcribe_btn.clicked.connect(self._start_transcription)
        btn_row.addWidget(self.transcribe_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(42)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setFixedWidth(90)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)
        main.addLayout(btn_row)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMinimumHeight(10)
        self.progress.setTextVisible(False)
        main.addWidget(self.progress)

        # Status label
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.status_lbl)

        # Result buttons (hidden until done)
        res_row = QHBoxLayout()
        res_row.setSpacing(10)

        self.open_folder_btn = QPushButton("📂  Open Folder")
        self.open_folder_btn.setMinimumHeight(36)
        self.open_folder_btn.setVisible(False)
        self.open_folder_btn.clicked.connect(self._open_folder)
        res_row.addWidget(self.open_folder_btn)

        self.open_txt_btn = QPushButton("📄  View TXT")
        self.open_txt_btn.setMinimumHeight(36)
        self.open_txt_btn.setVisible(False)
        self.open_txt_btn.clicked.connect(self._open_txt)
        res_row.addWidget(self.open_txt_btn)

        self.open_srt_btn = QPushButton("💬  View SRT")
        self.open_srt_btn.setMinimumHeight(36)
        self.open_srt_btn.setVisible(False)
        self.open_srt_btn.clicked.connect(self._open_srt)
        res_row.addWidget(self.open_srt_btn)

        main.addLayout(res_row)
        main.addStretch()

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("faster-whisper ready")
        self.setStatusBar(self.status_bar)

    # ── Theme ────────────────────────────────────────────────────────

    def _apply_theme(self):
        app = QApplication.instance()
        palette = app.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

        accent = "#4f8ef7"
        danger = "#e05252"

        if is_dark:
            bg = "#1e1e2e"
            surface = "#2a2a3e"
            border = "#3a3a5a"
            text = "#e0e0f0"
            sub = "#888"
        else:
            bg = "#f0f2f8"
            surface = "#ffffff"
            border = "#d0d4e8"
            text = "#1a1a2e"
            sub = "#666"

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {bg};
                color: {text};
                font-family: 'Segoe UI', 'SF Pro Text', system-ui, sans-serif;
                font-size: 13px;
            }}
            QLineEdit, QComboBox {{
                background: {surface};
                border: 1.5px solid {border};
                border-radius: 7px;
                padding: 5px 10px;
                color: {text};
            }}
            QLineEdit:focus, QComboBox:focus {{
                border-color: {accent};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QPushButton {{
                background: {surface};
                border: 1.5px solid {border};
                border-radius: 7px;
                padding: 5px 14px;
                color: {text};
            }}
            QPushButton:hover {{
                border-color: {accent};
                color: {accent};
            }}
            QPushButton:disabled {{
                opacity: 0.45;
            }}
            QPushButton#transcribe_btn {{
                background: {accent};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton#transcribe_btn:hover {{
                background: #3a7ae0;
                color: white;
            }}
            QPushButton#transcribe_btn:disabled {{
                background: {border};
                color: {sub};
            }}
            QPushButton#cancel_btn {{
                background: {danger};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton#cancel_btn:disabled {{
                background: {border};
                color: {sub};
            }}
            QProgressBar {{
                background: {border};
                border-radius: 5px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: {accent};
                border-radius: 5px;
            }}
            QLabel {{
                color: {text};
            }}
            QStatusBar {{
                color: {sub};
                font-size: 11px;
            }}
            QTextEdit {{
                background: {surface};
                border: 1.5px solid {border};
                border-radius: 7px;
                padding: 8px;
            }}
        """)

        self.transcribe_btn.setObjectName("transcribe_btn")
        self.cancel_btn.setObjectName("cancel_btn")
        # Re-apply so objectName styles take effect
        self.transcribe_btn.setStyleSheet("")
        self.cancel_btn.setStyleSheet("")

    # ── Settings persistence ─────────────────────────────────────────

    def _load_settings(self):
        lang = self._cfg.get("language", "Auto-detect")
        if lang in LANGUAGES:
            self.lang_box.setCurrentText(lang)
        model = self._cfg.get("model", DEFAULT_MODEL)
        if model in MODELS:
            self.model_box.setCurrentText(model)
        out = self._cfg.get("output_dir", DEFAULT_OUTPUT)
        self.out_input.setText(out)

    def _save_settings(self):
        self._cfg["language"] = self.lang_box.currentText()
        self._cfg["model"] = self.model_box.currentText()
        self._cfg["output_dir"] = self.out_input.text() or DEFAULT_OUTPUT
        save_config(self._cfg)

    # ── Slots ────────────────────────────────────────────────────────

    def _paste_url(self):
        clip = QApplication.clipboard().text().strip()
        if clip:
            self.url_input.setText(clip)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder", self.out_input.text() or DEFAULT_OUTPUT
        )
        if folder:
            self.out_input.setText(folder)

    def _start_transcription(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "No URL", "Please enter a YouTube URL.")
            return

        # Basic URL sanity check
        if not re.match(r"https?://", url):
            QMessageBox.warning(self, "Invalid URL", "URL must start with http:// or https://")
            return

        if not check_ffmpeg():
            dlg = FfmpegDialog(self)
            dlg.exec()
            return

        # Check imports
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "Missing dependency",
                                 "yt-dlp is not installed.\nRun: pip install yt-dlp")
            return
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "Missing dependency",
                                 "faster-whisper is not installed.\nRun: pip install faster-whisper")
            return

        self._save_settings()

        lang_key = self.lang_box.currentText()
        lang_code = LANGUAGES[lang_key]
        model_size = self.model_box.currentText()
        output_dir = self.out_input.text().strip() or DEFAULT_OUTPUT

        self._set_busy(True)
        self._result_txt = ""
        self._result_srt = ""
        self._hide_result_buttons()

        self._worker = TranscribeWorker(url, lang_code, model_size, output_dir)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.status_changed.connect(self._on_status)
        self._worker.progress_changed.connect(self.progress.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._thread.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self._cleanup_thread()
        self._set_busy(False)
        self.progress.setValue(0)
        self.status_lbl.setText("Cancelled.")

    def _on_status(self, msg: str):
        self.status_lbl.setText(msg)

    def _on_finished(self, txt_path: str, srt_path: str, title: str):
        self._result_txt = txt_path
        self._result_srt = srt_path
        self._cleanup_thread()
        self._set_busy(False)
        self._show_result_buttons()
        QMessageBox.information(
            self,
            "Transcription complete",
            f"✅  Done!\n\n"
            f"Title: {title}\n\n"
            f"TXT: {txt_path}\n"
            f"SRT: {srt_path}",
        )

    def _on_error(self, msg: str):
        self._cleanup_thread()
        self._set_busy(False)
        self.progress.setValue(0)
        self.status_lbl.setText("Error.")
        QMessageBox.critical(self, "Transcription failed", msg)

    # ── Result buttons ───────────────────────────────────────────────

    def _show_result_buttons(self):
        self.open_folder_btn.setVisible(True)
        self.open_txt_btn.setVisible(True)
        self.open_srt_btn.setVisible(True)

    def _hide_result_buttons(self):
        self.open_folder_btn.setVisible(False)
        self.open_txt_btn.setVisible(False)
        self.open_srt_btn.setVisible(False)

    def _open_folder(self):
        path = Path(self._result_txt).parent if self._result_txt else Path(self.out_input.text())
        self._open_path(str(path))

    def _open_txt(self):
        self._open_path(self._result_txt)

    def _open_srt(self):
        self._open_path(self._result_srt)

    @staticmethod
    def _open_path(path: str):
        if not path or not Path(path).exists():
            return
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    # ── Helpers ──────────────────────────────────────────────────────

    def _set_busy(self, busy: bool):
        self.transcribe_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.url_input.setEnabled(not busy)
        self.lang_box.setEnabled(not busy)
        self.model_box.setEnabled(not busy)
        self.out_input.setEnabled(not busy)
        if not busy:
            self.status_lbl.setText("Ready.")

    def _cleanup_thread(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
        self._worker = None

    def closeEvent(self, event):
        self._save_settings()
        if self._worker:
            self._worker.cancel()
        self._cleanup_thread()
        event.accept()


# ──────────────────────────────────────────────
#  SECTION 7: ENTRY POINT
# ──────────────────────────────────────────────

def main():
    # Enable HiDPI scaling on Qt6 (automatic, but explicitly set for Qt5 compat)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()