# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 zoott28354 and contributors

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import ctypes
from pathlib import Path

try:
    from PySide6.QtCore import QEvent, Qt, QTimer
    from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDragLeaveEvent, QDropEvent
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # helper functions remain importable before setup
    QApplication = None  # type: ignore[assignment]
    QMainWindow = object  # type: ignore[assignment,misc]


APP_DIR = Path(__file__).resolve().parent
QUANTIZER = APP_DIR / "quant_int8_convrot.py"
SUPPORTED_EXTENSIONS = {".safetensors", ".pth", ".pt", ".ckpt", ".bin"}

TRANSLATIONS = {
    "it": {
        "language": "Lingua:",
        "subtitle": "Quantizzazione INT8 + ConvRot per modelli ComfyUI, basata sul convertitore Comfy-Org.",
        "drop_default": "↓  TRASCINA QUI I MODELLI  ↓\n\n.safetensors  .pth  .pt  .ckpt  .bin",
        "drop_release": "RILASCIA PER AGGIUNGERE I MODELLI",
        "add_files": "Aggiungi file",
        "remove_selected": "Rimuovi selezionati",
        "clear": "Svuota",
        "model": "Modello",
        "size": "Dimensione",
        "state": "Stato",
        "output_folder": "Cartella output (vuota = accanto all'originale):",
        "browse": "Sfoglia",
        "dry_run": "Solo analisi (non scrive file)",
        "mse_clip": "MSE clip (sperimentale)",
        "downcast": "Riduci FP32 residui",
        "quality_report": "Salva report qualità .tsv",
        "delete_original": "Elimina definitivamente l'originale dopo la conversione riuscita",
        "min_gemm": "Min GEMM:",
        "cancel": "Annulla",
        "start": "Avvia conversione",
        "status_drop": "Trascina qui uno o più modelli",
        "status_release": "Rilascia i file nella finestra",
        "status_wait": "Attendi la fine della conversione prima di aggiungere altri file",
        "status_drop_error": "Impossibile leggere i file trascinati: {error}",
        "status_queue": "{count} modello/i in coda",
        "status_queue_rejected": "{count} modello/i in coda — {rejected} ignorati",
        "status_cancelling": "Annullamento in corso…",
        "status_cancelled": "Coda annullata",
        "status_done": "Operazione conclusa: {successes}/{total}",
        "state_waiting": "In attesa",
        "state_running": "In corso",
        "state_cancelled": "Annullato",
        "state_completed": "Completato",
        "state_analyzed": "Analizzato",
        "state_error": "Errore",
        "select_models": "Seleziona modelli ComfyUI",
        "models_filter": "Modelli",
        "all_files": "Tutti i file",
        "destination_folder": "Cartella di destinazione",
        "no_model_title": "Nessun modello",
        "no_model_message": "Aggiungi almeno un modello da convertire.",
        "invalid_value_title": "Valore non valido",
        "invalid_value_message": "Min GEMM deve essere un numero intero maggiore o uguale a zero.",
        "invalid_folder_title": "Cartella non valida",
        "invalid_folder_message": "La cartella di destinazione non esiste.",
        "active_conversion_title": "Conversione attiva",
        "active_conversion_message": "Interrompere la conversione e chiudere?",
        "missing_script_title": "Script mancante",
        "missing_script_message": "File non trovato:\n{path}",
        "log_queue_start": "Avvio coda ConvRot",
        "log_launch_error": "ERRORE avvio: {error}\n",
        "log_deleted_original": "Originale eliminato: {path}\n",
        "log_delete_error": "ATTENZIONE: impossibile eliminare l'originale {path}: {error}\n",
        "log_cancelled": "\nCoda annullata.\n",
        "log_done": "\nOperazione conclusa: {successes}/{total}.\n",
    },
    "en": {
        "language": "Language:",
        "subtitle": "INT8 + ConvRot quantization for ComfyUI models, based on the Comfy-Org converter.",
        "drop_default": "↓  DROP MODELS HERE  ↓\n\n.safetensors  .pth  .pt  .ckpt  .bin",
        "drop_release": "RELEASE TO ADD MODELS",
        "add_files": "Add files",
        "remove_selected": "Remove selected",
        "clear": "Clear",
        "model": "Model",
        "size": "Size",
        "state": "Status",
        "output_folder": "Output folder (blank = next to the original):",
        "browse": "Browse",
        "dry_run": "Analysis only (does not write files)",
        "mse_clip": "MSE clip (experimental)",
        "downcast": "Downcast remaining FP32",
        "quality_report": "Save .tsv quality report",
        "delete_original": "Permanently delete original after successful conversion",
        "min_gemm": "Min GEMM:",
        "cancel": "Cancel",
        "start": "Start conversion",
        "status_drop": "Drop one or more models here",
        "status_release": "Release the files in the window",
        "status_wait": "Wait for the conversion to finish before adding more files",
        "status_drop_error": "Unable to read dropped files: {error}",
        "status_queue": "{count} model(s) queued",
        "status_queue_rejected": "{count} model(s) queued — {rejected} ignored",
        "status_cancelling": "Cancelling…",
        "status_cancelled": "Queue cancelled",
        "status_done": "Operation complete: {successes}/{total}",
        "state_waiting": "Waiting",
        "state_running": "Running",
        "state_cancelled": "Cancelled",
        "state_completed": "Completed",
        "state_analyzed": "Analyzed",
        "state_error": "Error",
        "select_models": "Select ComfyUI models",
        "models_filter": "Models",
        "all_files": "All files",
        "destination_folder": "Destination folder",
        "no_model_title": "No model",
        "no_model_message": "Add at least one model to convert.",
        "invalid_value_title": "Invalid value",
        "invalid_value_message": "Min GEMM must be an integer greater than or equal to zero.",
        "invalid_folder_title": "Invalid folder",
        "invalid_folder_message": "The destination folder does not exist.",
        "active_conversion_title": "Conversion in progress",
        "active_conversion_message": "Stop the conversion and close the application?",
        "missing_script_title": "Missing script",
        "missing_script_message": "File not found:\n{path}",
        "log_queue_start": "Starting ConvRot queue",
        "log_launch_error": "LAUNCH ERROR: {error}\n",
        "log_deleted_original": "Deleted original: {path}\n",
        "log_delete_error": "WARNING: unable to delete original {path}: {error}\n",
        "log_cancelled": "\nQueue cancelled.\n",
        "log_done": "\nOperation complete: {successes}/{total}.\n",
    },
}


def enable_high_dpi_awareness() -> None:
    """Avoid Windows bitmap scaling (blur) and use the monitor's real DPI."""
    if os.name != "nt":
        return
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # Per-monitor V2
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor (Win 8.1)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def output_name(source: Path, mseclip: bool = False) -> str:
    """Apply the same output naming rule used by the upstream converter."""
    converted = re.sub(r"(?i)(bf16|fp16|fp32)", "int8_convrot", source.stem)
    if converted == source.stem:
        converted += "_int8_convrot"
    if mseclip:
        converted += "_mseclip"
    return converted + ".safetensors"


def output_path(source: Path, output_dir: Path | None, mseclip: bool = False) -> Path:
    return (output_dir or source.parent) / output_name(source, mseclip)


def numbered_output_path(base: Path, number: int) -> Path:
    if number == 0:
        return base
    return base.with_name(f"{base.stem} ({number}){base.suffix}")


def plan_output_paths(
    sources: list[Path], output_dir: Path | None, write_report: bool, mseclip: bool = False
) -> dict[Path, Path]:
    normalize = lambda path: os.path.normcase(os.path.realpath(os.path.abspath(path)))
    reserved = {normalize(source) for source in sources}
    planned: dict[Path, Path] = {}
    for source in sources:
        base = output_path(source, output_dir, mseclip)
        number = 0
        while True:
            destination = numbered_output_path(base, number)
            artifacts = [destination]
            if write_report:
                artifacts.append(destination.with_suffix(".quality.tsv"))
            if not any(normalize(path) in reserved or path.exists() for path in artifacts):
                break
            number += 1
        planned[source] = destination
        reserved.update(normalize(path) for path in artifacts)
    return planned


def delete_original_after_conversion(source: Path, destination: Path) -> None:
    """Delete a source only after validating a distinct, non-empty converted file."""
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=True)
    if source == destination or os.path.samefile(source, destination):
        raise ValueError("source and destination refer to the same file")
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise ValueError("converted output is missing or empty")
    source.unlink()


def build_command(
    source: Path,
    destination: Path | None,
    *,
    dry_run: bool,
    min_gemm: int,
    mseclip: bool,
    downcast_fp32: bool,
    report_path: Path | None,
) -> list[str]:
    command = [sys.executable, "-u", str(QUANTIZER), str(source)]
    if not dry_run and destination is not None:
        command.append(str(destination))
    if dry_run:
        command.append("--dry-run")
    command.extend(["--min-gemm", str(min_gemm)])
    if mseclip:
        command.append("--mseclip")
    if downcast_fp32:
        command.append("--downcast-fp32")
    if report_path is not None and not dry_run:
        command.extend(["--verify-report", str(report_path)])
    return command


class ConvRotApp(QMainWindow):
    BG = "#0b1120"
    PANEL = "#111827"
    PANEL_2 = "#182235"
    BORDER = "#2b3a52"
    TEXT = "#f8fafc"
    MUTED = "#94a3b8"
    ACCENT = "#38bdf8"
    DANGER = "#fb7185"

    def __init__(self) -> None:
        if QApplication is None:
            raise SystemExit("Missing dependency: PySide6. Run setup.bat first.")
        super().__init__()
        self.setWindowTitle("ComfyUI ConvRot Converter")
        self.resize(1080, 860)
        self.setMinimumSize(820, 680)
        self.setAcceptDrops(True)

        self.files: list[Path] = []
        self.file_states: dict[Path, str] = {}
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.current_language = "en"
        self.status_context: tuple[str, dict[str, object]] | None = ("status_drop", {})

        self._build_ui()
        self._configure_style()
        self.event_timer = QTimer(self)
        self.event_timer.timeout.connect(self._poll_events)
        self.event_timer.start(100)

    def _t(self, key: str, **values: object) -> str:
        return TRANSLATIONS[self.current_language][key].format(**values)

    def _set_status(self, key: str, **values: object) -> None:
        self.status_context = (key, values)
        self.status_label.setText(self._t(key, **values))

    def _set_literal_status(self, text: str) -> None:
        self.status_context = None
        self.status_label.setText(text)

    def _configure_style(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget#central {{ background: {self.BG}; color: {self.TEXT}; }}
            QWidget {{ font-family: 'Segoe UI'; font-size: 10pt; color: {self.TEXT}; }}
            QLabel#title {{ font-size: 24pt; font-weight: 700; }}
            QLabel#subtitle, QLabel#muted, QLabel#status {{ color: {self.MUTED}; }}
            QFrame#drop, QFrame#options {{
                background: {self.PANEL}; border: 1px solid {self.BORDER}; border-radius: 12px;
            }}
            QLabel#dropText {{ color: {self.ACCENT}; font-size: 13pt; font-weight: 650; }}
            QPushButton {{
                background: {self.PANEL_2}; border: 1px solid {self.BORDER}; border-radius: 8px;
                padding: 8px 14px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #25334a; border-color: #46617f; }}
            QPushButton:disabled {{ color: #64748b; background: #131c2c; border-color: #223047; }}
            QPushButton#start {{ background: {self.ACCENT}; color: #082f49; border: none; padding: 10px 18px; }}
            QPushButton#start:hover {{ background: #7dd3fc; }}
            QPushButton#cancel {{ background: #3b1727; color: #fecdd3; border-color: #7f1d3b; }}
            QLineEdit, QSpinBox, QComboBox {{
                background: {self.PANEL_2}; border: 1px solid {self.BORDER}; border-radius: 7px;
                padding: 7px 9px; selection-background-color: #0369a1;
            }}
            QComboBox::drop-down {{ border: 0; width: 24px; }}
            QCheckBox {{ spacing: 8px; }}
            QCheckBox::indicator {{ width: 17px; height: 17px; }}
            QTableWidget {{
                background: {self.PANEL}; alternate-background-color: #142033; border: 1px solid {self.BORDER};
                border-radius: 10px; gridline-color: #24334a; selection-background-color: #075985;
            }}
            QHeaderView::section {{
                background: {self.PANEL_2}; color: {self.TEXT}; border: none; border-bottom: 1px solid {self.BORDER};
                padding: 8px; font-weight: 650;
            }}
            QPlainTextEdit {{
                background: #070d18; color: #d1d5db; border: 1px solid {self.BORDER};
                border-radius: 10px; padding: 8px; font-family: 'Cascadia Mono', Consolas; font-size: 9pt;
            }}
            QSplitter::handle {{ background: transparent; height: 8px; }}
            QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
            QScrollBar::handle:vertical {{ background: #334155; border-radius: 5px; min-height: 28px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("central")
        self.setCentralWidget(central)
        shell = QVBoxLayout(central)
        shell.setContentsMargins(24, 20, 24, 20)
        shell.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("ConvRot Converter")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        self.language_label = QLabel(self._t("language"))
        self.language_label.setObjectName("muted")
        header.addWidget(self.language_label)
        self.language_switch = QComboBox()
        self.language_switch.addItems(["ENG", "ITA"])
        self.language_switch.setFixedWidth(82)
        self.language_switch.currentTextChanged.connect(self._change_language)
        header.addWidget(self.language_switch)
        shell.addLayout(header)

        self.subtitle_label = QLabel(self._t("subtitle"))
        self.subtitle_label.setObjectName("subtitle")
        self.subtitle_label.setWordWrap(True)
        shell.addWidget(self.subtitle_label)

        self.drop_frame = QFrame()
        self.drop_frame.setObjectName("drop")
        self.drop_frame.setMinimumHeight(112)
        self.drop_frame.setCursor(Qt.CursorShape.PointingHandCursor)
        drop_layout = QVBoxLayout(self.drop_frame)
        self.drop_label = QLabel(self._t("drop_default"))
        self.drop_label.setObjectName("dropText")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        drop_layout.addWidget(self.drop_label)
        self.drop_frame.installEventFilter(self)
        shell.addWidget(self.drop_frame)

        queue_buttons = QHBoxLayout()
        self.add_button = QPushButton(self._t("add_files"))
        self.add_button.clicked.connect(self._browse_files)
        self.remove_button = QPushButton(self._t("remove_selected"))
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button = QPushButton(self._t("clear"))
        self.clear_button.clicked.connect(self._clear_files)
        queue_buttons.addWidget(self.add_button)
        queue_buttons.addWidget(self.remove_button)
        queue_buttons.addWidget(self.clear_button)
        queue_buttons.addStretch()
        shell.addLayout(queue_buttons)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([self._t("model"), self._t("size"), self._t("state")])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        table_header = self.table.horizontalHeader()
        table_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        options = QFrame()
        options.setObjectName("options")
        options_layout = QGridLayout(options)
        options_layout.setContentsMargins(16, 14, 16, 14)
        options_layout.setHorizontalSpacing(18)
        options_layout.setVerticalSpacing(9)
        self.output_label = QLabel(self._t("output_folder"))
        options_layout.addWidget(self.output_label, 0, 0, 1, 3)
        self.output_entry = QLineEdit()
        options_layout.addWidget(self.output_entry, 1, 0, 1, 2)
        self.browse_output_button = QPushButton(self._t("browse"))
        self.browse_output_button.clicked.connect(self._browse_output)
        options_layout.addWidget(self.browse_output_button, 1, 2)
        self.dry_run_check = QCheckBox(self._t("dry_run"))
        self.mse_check = QCheckBox(self._t("mse_clip"))
        self.downcast_check = QCheckBox(self._t("downcast"))
        self.report_check = QCheckBox(self._t("quality_report"))
        self.report_check.setChecked(True)
        self.delete_original_check = QCheckBox(self._t("delete_original"))
        self.dry_run_check.toggled.connect(self._dry_run_changed)
        options_layout.addWidget(self.dry_run_check, 2, 0)
        options_layout.addWidget(self.mse_check, 2, 1)
        options_layout.addWidget(self.downcast_check, 3, 0)
        options_layout.addWidget(self.report_check, 3, 1)
        options_layout.addWidget(self.delete_original_check, 4, 0, 1, 2)
        self.min_gemm_label = QLabel(self._t("min_gemm"))
        options_layout.addWidget(self.min_gemm_label, 2, 2, alignment=Qt.AlignmentFlag.AlignRight)
        self.min_gemm_spin = QSpinBox()
        self.min_gemm_spin.setRange(0, 8192)
        self.min_gemm_spin.setSingleStep(16)
        self.min_gemm_spin.setValue(256)
        self.min_gemm_spin.setFixedWidth(105)
        options_layout.addWidget(self.min_gemm_spin, 3, 2, alignment=Qt.AlignmentFlag.AlignRight)
        options_layout.setColumnStretch(0, 1)
        options_layout.setColumnStretch(1, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.table)
        lower = QWidget()
        lower_layout = QVBoxLayout(lower)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(12)
        lower_layout.addWidget(options)
        action = QHBoxLayout()
        self.status_label = QLabel(self._t("status_drop"))
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        action.addWidget(self.status_label, 1)
        self.start_button = QPushButton(self._t("start"))
        self.start_button.setObjectName("start")
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton(self._t("cancel"))
        self.cancel_button.setObjectName("cancel")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        action.addWidget(self.start_button)
        action.addWidget(self.cancel_button)
        lower_layout.addLayout(action)
        lower_layout.addWidget(self.log, 1)
        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([230, 390])
        shell.addWidget(splitter, 1)

    def _change_language(self, language: str) -> None:
        self.current_language = "en" if language == "ENG" else "it"
        self._apply_language()

    def _apply_language(self) -> None:
        self.language_label.setText(self._t("language"))
        self.subtitle_label.setText(self._t("subtitle"))
        self.add_button.setText(self._t("add_files"))
        self.remove_button.setText(self._t("remove_selected"))
        self.clear_button.setText(self._t("clear"))
        self.output_label.setText(self._t("output_folder"))
        self.browse_output_button.setText(self._t("browse"))
        self.dry_run_check.setText(self._t("dry_run"))
        self.mse_check.setText(self._t("mse_clip"))
        self.downcast_check.setText(self._t("downcast"))
        self.report_check.setText(self._t("quality_report"))
        self.delete_original_check.setText(self._t("delete_original"))
        self.min_gemm_label.setText(self._t("min_gemm"))
        self.cancel_button.setText(self._t("cancel"))
        self.start_button.setText(self._t("start"))
        self.table.setHorizontalHeaderLabels([self._t("model"), self._t("size"), self._t("state")])
        self._reset_drop_banner()
        for path, state_key in self.file_states.items():
            row = self._row_for_path(path)
            if row is not None:
                self.table.item(row, 2).setText(self._t(state_key))
        if self.status_context is not None:
            key, values = self.status_context
            self.status_label.setText(self._t(key, **values))

    def _worker_active(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def _dry_run_changed(self, checked: bool) -> None:
        if checked:
            self.delete_original_check.setChecked(False)
        self.delete_original_check.setEnabled(not checked)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched is self.drop_frame and event.type() == QEvent.Type.MouseButtonRelease:
            if not self._worker_active():
                self._browse_files()
            return True
        return super().eventFilter(watched, event)

    def _drop_paths(self, event: QDropEvent | QDragEnterEvent) -> list[Path]:
        return [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        paths = self._drop_paths(event)
        if paths and not self._worker_active():
            event.acceptProposedAction()
            self.drop_frame.setStyleSheet("QFrame#drop { background: #075985; border-color: #38bdf8; }")
            self.drop_label.setStyleSheet("color: #e0f2fe;")
            self.drop_label.setText(self._t("drop_release"))
            self._set_status("status_release")
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._reset_drop_banner()
        if not self._worker_active():
            self._set_status("status_queue", count=len(self.files)) if self.files else self._set_status("status_drop")
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        self._reset_drop_banner()
        if self._worker_active():
            self._set_status("status_wait")
            event.ignore()
            return
        self._add_files(self._drop_paths(event))
        event.acceptProposedAction()

    def _reset_drop_banner(self) -> None:
        self.drop_frame.setStyleSheet("")
        self.drop_label.setStyleSheet("")
        self.drop_label.setText(self._t("drop_default"))

    @staticmethod
    def _size_label(path: Path) -> str:
        size = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return ""

    def _browse_files(self) -> None:
        model_filter = f"{self._t('models_filter')} (*.safetensors *.pth *.pt *.ckpt *.bin)"
        all_filter = f"{self._t('all_files')} (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, self._t("select_models"), "", f"{model_filter};;{all_filter}")
        self._add_files([Path(path) for path in paths])

    def _add_files(self, paths: list[Path]) -> None:
        rejected = 0
        for path in paths:
            path = path.resolve()
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                rejected += 1
                continue
            if path not in self.files:
                self.files.append(path)
                self.file_states[path] = "state_waiting"
                row = self.table.rowCount()
                self.table.insertRow(row)
                name_item = QTableWidgetItem(path.name)
                name_item.setData(Qt.ItemDataRole.UserRole, str(path))
                name_item.setToolTip(str(path))
                size_item = QTableWidgetItem(self._size_label(path))
                size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                state_item = QTableWidgetItem(self._t("state_waiting"))
                state_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, 0, name_item)
                self.table.setItem(row, 1, size_item)
                self.table.setItem(row, 2, state_item)
        if rejected:
            self._set_status("status_queue_rejected", count=len(self.files), rejected=rejected)
        else:
            self._set_status("status_queue", count=len(self.files))

    def _row_for_path(self, path: Path) -> int | None:
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == str(path):
                return row
        return None

    def _remove_selected(self) -> None:
        if self._worker_active():
            return
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            path = Path(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            if path in self.files:
                self.files.remove(path)
            self.file_states.pop(path, None)
            self.table.removeRow(row)
        self._set_status("status_queue", count=len(self.files)) if self.files else self._set_status("status_drop")

    def _clear_files(self) -> None:
        if self._worker_active():
            return
        self.files.clear()
        self.file_states.clear()
        self.table.setRowCount(0)
        self._set_status("status_drop")

    def _browse_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, self._t("destination_folder"))
        if chosen:
            self.output_entry.setText(chosen)

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _start(self) -> None:
        if not self.files:
            QMessageBox.information(self, self._t("no_model_title"), self._t("no_model_message"))
            return
        min_gemm = self.min_gemm_spin.value()
        output_text = self.output_entry.text().strip()
        out_dir = Path(output_text).expanduser().resolve() if output_text else None
        if out_dir and not out_dir.is_dir():
            QMessageBox.critical(self, self._t("invalid_folder_title"), self._t("invalid_folder_message"))
            return

        self.cancel_requested = False
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.drop_frame.setCursor(Qt.CursorShape.ArrowCursor)
        self._append_log("\n" + "=" * 72 + "\n" + self._t("log_queue_start") + "\n")
        snapshot = list(self.files)
        dry_run = self.dry_run_check.isChecked()
        write_report = self.report_check.isChecked()
        mseclip = self.mse_check.isChecked()
        delete_original = self.delete_original_check.isChecked()
        destinations = {} if dry_run else plan_output_paths(snapshot, out_dir, write_report, mseclip)
        options = (
            destinations, dry_run, min_gemm, mseclip,
            self.downcast_check.isChecked(), write_report, delete_original,
        )
        self.worker = threading.Thread(target=self._run_queue, args=(snapshot, options), daemon=True)
        self.worker.start()

    def _run_queue(
        self,
        files: list[Path],
        options: tuple[dict[Path, Path], bool, int, bool, bool, bool, bool],
    ) -> None:
        destinations, dry_run, min_gemm, mseclip, downcast, write_report, delete_original = options
        successes = 0
        for index, source in enumerate(files, start=1):
            if self.cancel_requested:
                break
            destination = None if dry_run else destinations[source]
            report = destination.with_suffix(".quality.tsv") if write_report and destination else None
            command = build_command(
                source, destination, dry_run=dry_run, min_gemm=min_gemm,
                mseclip=mseclip, downcast_fp32=downcast, report_path=report,
            )
            self.events.put(("state", (source, "state_running")))
            self.events.put(("status", f"{index}/{len(files)} — {source.name}"))
            self.events.put(("log", f"\n>>> {source}\n"))
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=APP_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.events.put(("log", line))
                return_code = self.process.wait()
            except Exception as exc:
                self.events.put(("log_key", ("log_launch_error", {"error": exc})))
                return_code = -1
            finally:
                self.process = None
            if self.cancel_requested:
                self.events.put(("state", (source, "state_cancelled")))
                break
            if return_code == 0:
                successes += 1
                if delete_original and destination is not None:
                    try:
                        delete_original_after_conversion(source, destination)
                        self.events.put(("log_key", ("log_deleted_original", {"path": source})))
                    except (OSError, ValueError) as exc:
                        self.events.put(("log_key", ("log_delete_error", {"path": source, "error": exc})))
                self.events.put(("state", (source, "state_analyzed" if dry_run else "state_completed")))
            else:
                self.events.put(("state", (source, "state_error")))
        self.events.put(("done", (successes, len(files), self.cancel_requested)))

    def _cancel(self) -> None:
        self.cancel_requested = True
        self._set_status("status_cancelling")
        process = self.process
        if process and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "log_key":
                    key, values = payload  # type: ignore[misc]
                    self._append_log(self._t(key, **values))
                elif kind == "status":
                    self._set_literal_status(str(payload))
                elif kind == "state":
                    source, state_key = payload  # type: ignore[misc]
                    self.file_states[source] = state_key
                    row = self._row_for_path(source)
                    if row is not None:
                        self.table.item(row, 2).setText(self._t(state_key))
                elif kind == "done":
                    successes, total, cancelled = payload  # type: ignore[misc]
                    self.start_button.setEnabled(True)
                    self.cancel_button.setEnabled(False)
                    self.drop_frame.setCursor(Qt.CursorShape.PointingHandCursor)
                    if cancelled:
                        self._set_status("status_cancelled")
                        self._append_log(self._t("log_cancelled"))
                    else:
                        self._set_status("status_done", successes=successes, total=total)
                        self._append_log(self._t("log_done", successes=successes, total=total))
        except queue.Empty:
            pass

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker_active():
            answer = QMessageBox.question(
                self, self._t("active_conversion_title"), self._t("active_conversion_message"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._cancel()
        event.accept()


def main() -> int:
    if QApplication is None:
        raise SystemExit("Missing dependency: PySide6. Run setup.bat first.")
    enable_high_dpi_awareness()
    app = QApplication(sys.argv)
    app.setApplicationName("ComfyUI ConvRot Converter")
    app.setStyle("Fusion")
    window = ConvRotApp()
    if not QUANTIZER.is_file():
        QMessageBox.critical(
            window, window._t("missing_script_title"),
            window._t("missing_script_message", path=QUANTIZER),
        )
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
