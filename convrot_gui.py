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
from dataclasses import dataclass
from pathlib import Path

try:
    from PySide6.QtCore import QEvent, Qt, QTimer
    from PySide6.QtGui import QColor, QCloseEvent, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
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
APP_VERSION = "1.0.0"
QUANTIZER = APP_DIR / "quant_int8_convrot.py"
APP_ICON = APP_DIR / "assets" / "app-icon.ico"
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
        "compatibility": "Compatibilità",
        "compatibility_title": "Controllo automatico compatibilità",
        "compatibility_select": "Seleziona un modello per vedere il risultato dell'analisi.",
        "compatibility_checking": "Analisi in corso…",
        "compatibility_ready": "Pronto per la conversione",
        "compatibility_limited": "Convertibile, vantaggio limitato",
        "compatibility_unsafe": "Non convertire",
        "compatibility_failed": "Non compatibile",
        "compatibility_details": "Sorgente {dtype} • preset {preset} • {layers} livelli • {params:.2f}B parametri • copertura {coverage:.1f}%",
        "compatibility_ready_note": "La struttura è tecnicamente compatibile. Il report qualità verificherà il risultato dopo la conversione.",
        "compatibility_limited_note": "La conversione è possibile, ma interessa solo una parte ridotta del modello.",
        "compatibility_unsafe_note": "La sorgente risulta già quantizzata: una seconda quantizzazione può peggiorare sensibilmente la qualità.",
        "compatibility_failed_note": "Il controllo automatico non ha trovato livelli compatibili o ha restituito un errore: {error}",
        "status_preflight": "Controllo compatibilità di {count} modello/i…",
        "about": "Informazioni",
        "about_description": "Conversione drag-and-drop di modelli ComfyUI in INT8 + ConvRot.",
        "about_created_by": "Creato da <a href=\"https://github.com/zoott28354\">zoott28354</a>",
        "about_project": "<a href=\"https://github.com/zoott28354/Comfyui-convrot-converter\">Progetto su GitHub</a>",
        "about_based_on": "Convertitore basato sugli strumenti open source di <a href=\"https://github.com/Comfy-Org/comfy-model-tools\">Comfy-Org</a>.",
        "about_license": "Software libero distribuito con licenza GNU GPL-3.0.",
        "close": "Chiudi",
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
        "compatibility": "Compatibility",
        "compatibility_title": "Automatic compatibility check",
        "compatibility_select": "Select a model to view its analysis result.",
        "compatibility_checking": "Checking…",
        "compatibility_ready": "Ready to convert",
        "compatibility_limited": "Convertible, limited benefit",
        "compatibility_unsafe": "Do not convert",
        "compatibility_failed": "Not compatible",
        "compatibility_details": "{dtype} source • {preset} preset • {layers} layers • {params:.2f}B parameters • {coverage:.1f}% coverage",
        "compatibility_ready_note": "The structure is technically compatible. The quality report will verify the result after conversion.",
        "compatibility_limited_note": "Conversion is possible, but it affects only a small part of the model.",
        "compatibility_unsafe_note": "The source is already quantized: a second quantization can significantly reduce quality.",
        "compatibility_failed_note": "The automatic check found no compatible layers or returned an error: {error}",
        "status_preflight": "Checking compatibility for {count} model(s)…",
        "about": "About",
        "about_description": "Drag-and-drop conversion of ComfyUI models to INT8 + ConvRot.",
        "about_created_by": "Created by <a href=\"https://github.com/zoott28354\">zoott28354</a>",
        "about_project": "<a href=\"https://github.com/zoott28354/Comfyui-convrot-converter\">Project on GitHub</a>",
        "about_based_on": "Converter based on the open-source tools by <a href=\"https://github.com/Comfy-Org/comfy-model-tools\">Comfy-Org</a>.",
        "about_license": "Free software released under the GNU GPL-3.0 license.",
        "close": "Close",
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


@dataclass(frozen=True)
class CompatibilityResult:
    verdict: str
    source_dtype: str = "unknown"
    preset: str = "unknown"
    quantized_layers: int = 0
    quantized_params_b: float = 0.0
    source_elements_b: float = 0.0
    error: str = ""

    @property
    def coverage(self) -> float:
        if self.source_elements_b <= 0:
            return 0.0
        return min(100.0, self.quantized_params_b / self.source_elements_b * 100.0)


def parse_compatibility_output(output: str, return_code: int) -> CompatibilityResult:
    """Turn quantizer dry-run output into a conservative technical verdict."""
    dtype_rows = re.findall(
        r"^\s+(BF16|F16|F32|F8_E4M3|F8_E5M2|I8)\s+([0-9.]+)B elements\s+([0-9.]+)%",
        output,
        flags=re.MULTILINE,
    )
    source_elements = sum(float(elements) for _, elements, _ in dtype_rows)
    source_dtype = max(dtype_rows, key=lambda row: float(row[2]))[0] if dtype_rows else "unknown"
    preset_match = re.search(r"^layer-selection preset:\s*(\S+)", output, flags=re.MULTILINE)
    layers_match = re.search(r"^QUANTIZE\s+(\d+)\s+layers", output, flags=re.MULTILINE)
    params_match = re.search(r"quantized params:\s*([0-9.]+)B", output)
    preset = preset_match.group(1) if preset_match else "unknown"
    layers = int(layers_match.group(1)) if layers_match else 0
    params = float(params_match.group(1)) if params_match else 0.0

    if return_code != 0:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return CompatibilityResult(
            "failed", source_dtype, preset, layers, params, source_elements,
            lines[-1] if lines else f"exit code {return_code}",
        )
    if "already predominantly FP8" in output or "already predominantly INT8" in output \
            or source_dtype.startswith("F8") or source_dtype == "I8":
        return CompatibilityResult("unsafe", source_dtype, preset, layers, params, source_elements)
    if layers <= 0 or params <= 0:
        return CompatibilityResult(
            "failed", source_dtype, preset, layers, params, source_elements,
            "no eligible INT8 ConvRot layers",
        )

    coverage = params / source_elements * 100.0 if source_elements else 0.0
    verdict = "ready" if coverage >= 25.0 else "limited"
    return CompatibilityResult(verdict, source_dtype, preset, layers, params, source_elements)


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
        self.compatibility_results: dict[Path, CompatibilityResult] = {}
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.preflight_thread: threading.Thread | None = None
        self.preflight_process: subprocess.Popen[str] | None = None
        self.preflight_generation = 0
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
            QMainWindow, QWidget#central, QDialog {{ background: {self.BG}; color: {self.TEXT}; }}
            QWidget {{ font-family: 'Segoe UI'; font-size: 10pt; color: {self.TEXT}; }}
            QLabel#title {{ font-size: 24pt; font-weight: 700; }}
            QLabel#aboutTitle {{ font-size: 17pt; font-weight: 700; }}
            QLabel#subtitle, QLabel#muted, QLabel#status {{ color: {self.MUTED}; }}
            QFrame#drop, QFrame#options, QFrame#compatibilityPanel {{
                background: {self.PANEL}; border: 1px solid {self.BORDER}; border-radius: 12px;
            }}
            QLabel#dropText {{ color: {self.ACCENT}; font-size: 13pt; font-weight: 650; }}
            QLabel#compatibilitySummary {{ font-size: 13pt; font-weight: 700; }}
            QPushButton {{
                background: {self.PANEL_2}; border: 1px solid {self.BORDER}; border-radius: 8px;
                padding: 8px 14px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #25334a; border-color: #46617f; }}
            QPushButton:disabled {{ color: #64748b; background: #131c2c; border-color: #223047; }}
            QPushButton#start {{ background: {self.ACCENT}; color: #082f49; border: none; padding: 10px 18px; }}
            QPushButton#start:hover {{ background: #7dd3fc; }}
            QPushButton#cancel {{ background: #3b1727; color: #fecdd3; border-color: #7f1d3b; }}
            QPushButton#about {{ font-size: 12pt; padding: 0; border-radius: 18px; }}
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
        self.about_button = QPushButton("?")
        self.about_button.setObjectName("about")
        self.about_button.setFixedSize(36, 36)
        self.about_button.setToolTip(self._t("about"))
        self.about_button.clicked.connect(self._show_about)
        header.addWidget(self.about_button)
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

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            self._t("model"), self._t("size"), self._t("state"), self._t("compatibility")
        ])
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
        table_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._update_compatibility_panel)

        self.compatibility_panel = QFrame()
        self.compatibility_panel.setObjectName("compatibilityPanel")
        compatibility_layout = QVBoxLayout(self.compatibility_panel)
        compatibility_layout.setContentsMargins(16, 12, 16, 12)
        compatibility_layout.setSpacing(4)
        self.compatibility_title_label = QLabel(self._t("compatibility_title"))
        self.compatibility_title_label.setObjectName("muted")
        self.compatibility_summary_label = QLabel(self._t("compatibility_select"))
        self.compatibility_summary_label.setObjectName("compatibilitySummary")
        self.compatibility_details_label = QLabel("")
        self.compatibility_details_label.setObjectName("muted")
        self.compatibility_details_label.setWordWrap(True)
        compatibility_layout.addWidget(self.compatibility_title_label)
        compatibility_layout.addWidget(self.compatibility_summary_label)
        compatibility_layout.addWidget(self.compatibility_details_label)

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
        self.min_gemm_spin.valueChanged.connect(self._preflight_settings_changed)
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
        lower_layout.addWidget(self.compatibility_panel)
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
        self.about_button.setToolTip(self._t("about"))
        self.compatibility_title_label.setText(self._t("compatibility_title"))
        self.table.setHorizontalHeaderLabels([
            self._t("model"), self._t("size"), self._t("state"), self._t("compatibility")
        ])
        self._reset_drop_banner()
        for path, state_key in self.file_states.items():
            row = self._row_for_path(path)
            if row is not None:
                self.table.item(row, 2).setText(self._t(state_key))
                self._set_compatibility_cell(row, self.compatibility_results.get(path))
        self._update_compatibility_panel()
        if self.status_context is not None:
            key, values = self.status_context
            self.status_label.setText(self._t(key, **values))

    def _show_about(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._t("about"))
        dialog.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        dialog.setModal(True)
        dialog.setMinimumWidth(470)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)

        top = QHBoxLayout()
        icon_label = QLabel()
        if APP_ICON.is_file():
            icon_label.setPixmap(QIcon(str(APP_ICON)).pixmap(80, 80))
        icon_label.setFixedSize(88, 88)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(icon_label)

        heading = QVBoxLayout()
        name_label = QLabel("ComfyUI ConvRot Converter")
        name_label.setObjectName("aboutTitle")
        version_label = QLabel(f"Version {APP_VERSION}")
        version_label.setObjectName("muted")
        description_label = QLabel(self._t("about_description"))
        description_label.setObjectName("muted")
        description_label.setWordWrap(True)
        heading.addWidget(name_label)
        heading.addWidget(version_label)
        heading.addWidget(description_label)
        heading.addStretch()
        top.addLayout(heading, 1)
        layout.addLayout(top)

        details_html = (
            f"{self._t('about_created_by')}<br>"
            f"{self._t('about_project')}<br><br>"
            f"{self._t('about_based_on')}<br>"
            f"{self._t('about_license')}<br><br>"
            "© 2026 zoott28354 and contributors"
        )
        details_html = details_html.replace('<a href=', f'<a style="color:{self.ACCENT};" href=')
        details = QLabel(details_html)
        details.setTextFormat(Qt.TextFormat.RichText)
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        details.setOpenExternalLinks(True)
        details.setWordWrap(True)
        layout.addWidget(details)

        close_button = QPushButton(self._t("close"))
        close_button.clicked.connect(dialog.accept)
        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(close_button)
        layout.addLayout(button_row)
        dialog.exec()

    def _worker_active(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def _preflight_active(self) -> bool:
        return bool(self.preflight_thread and self.preflight_thread.is_alive())

    def _compatibility_text(self, result: CompatibilityResult | None) -> str:
        if result is None:
            return self._t("compatibility_checking")
        return self._t(f"compatibility_{result.verdict}")

    def _set_compatibility_cell(self, row: int, result: CompatibilityResult | None) -> None:
        item = self.table.item(row, 3)
        if item is None:
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, item)
        item.setText(self._compatibility_text(result))
        colors = {
            None: self.MUTED,
            "ready": "#34d399",
            "limited": "#fbbf24",
            "unsafe": self.DANGER,
            "failed": self.DANGER,
        }
        item.setForeground(QColor(colors[result.verdict if result else None]))

    def _selected_path(self) -> Path | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        value = self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        return Path(value) if value else None

    def _update_compatibility_panel(self) -> None:
        path = self._selected_path()
        if path is None:
            self.compatibility_summary_label.setText(self._t("compatibility_select"))
            self.compatibility_summary_label.setStyleSheet(f"color: {self.MUTED};")
            self.compatibility_details_label.clear()
            return
        result = self.compatibility_results.get(path)
        if result is None:
            self.compatibility_summary_label.setText(self._t("compatibility_checking"))
            self.compatibility_summary_label.setStyleSheet(f"color: {self.ACCENT};")
            self.compatibility_details_label.setText(path.name)
            return

        colors = {"ready": "#34d399", "limited": "#fbbf24", "unsafe": self.DANGER, "failed": self.DANGER}
        self.compatibility_summary_label.setText(f"{path.name} — {self._compatibility_text(result)}")
        self.compatibility_summary_label.setStyleSheet(f"color: {colors[result.verdict]};")
        if result.verdict in ("ready", "limited"):
            details = self._t(
                "compatibility_details", dtype=result.source_dtype, preset=result.preset,
                layers=result.quantized_layers, params=result.quantized_params_b,
                coverage=result.coverage,
            )
            note = self._t(f"compatibility_{result.verdict}_note")
        elif result.verdict == "unsafe":
            details = self._t(
                "compatibility_details", dtype=result.source_dtype, preset=result.preset,
                layers=result.quantized_layers, params=result.quantized_params_b,
                coverage=result.coverage,
            )
            note = self._t("compatibility_unsafe_note")
        else:
            details = ""
            note = self._t("compatibility_failed_note", error=result.error or "unknown error")
        self.compatibility_details_label.setText(f"{details}\n{note}".strip())

    def _dry_run_changed(self, checked: bool) -> None:
        if checked:
            self.delete_original_check.setChecked(False)
        self.delete_original_check.setEnabled(not checked)

    def _preflight_settings_changed(self, _value: int) -> None:
        if not self.files or self._worker_active():
            return
        self._invalidate_preflight()
        self.compatibility_results.clear()
        for row in range(self.table.rowCount()):
            self._set_compatibility_cell(row, None)
        self._update_compatibility_panel()
        self._start_preflight()

    def _invalidate_preflight(self) -> None:
        """Discard an obsolete check and stop its current subprocess."""
        self.preflight_generation += 1
        process = self.preflight_process
        if process and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

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

    def _start_preflight(self) -> None:
        if self._worker_active() or self._preflight_active():
            return
        pending = [path for path in self.files if path not in self.compatibility_results]
        if not pending:
            self.start_button.setEnabled(True)
            return
        self.start_button.setEnabled(False)
        self._set_status("status_preflight", count=len(pending))
        min_gemm = self.min_gemm_spin.value()
        generation = self.preflight_generation
        self.preflight_thread = threading.Thread(
            target=self._run_preflight, args=(pending, min_gemm, generation), daemon=True,
        )
        self.preflight_thread.start()

    def _run_preflight(self, files: list[Path], min_gemm: int, generation: int) -> None:
        for source in files:
            if generation != self.preflight_generation:
                break
            if source not in self.files:
                continue
            command = build_command(
                source, None, dry_run=True, min_gemm=min_gemm,
                mseclip=False, downcast_fp32=False, report_path=None,
            )
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            try:
                self.preflight_process = subprocess.Popen(
                    command,
                    cwd=APP_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                output, _ = self.preflight_process.communicate()
                return_code = self.preflight_process.returncode
                result = parse_compatibility_output(output, return_code)
            except Exception as exc:
                result = CompatibilityResult("failed", error=str(exc))
            finally:
                self.preflight_process = None
            self.events.put(("compatibility", (source, result, generation)))
        self.events.put(("preflight_done", generation))

    def _browse_files(self) -> None:
        model_filter = f"{self._t('models_filter')} (*.safetensors *.pth *.pt *.ckpt *.bin)"
        all_filter = f"{self._t('all_files')} (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, self._t("select_models"), "", f"{model_filter};;{all_filter}")
        self._add_files([Path(path) for path in paths])

    def _add_files(self, paths: list[Path]) -> None:
        rejected = 0
        added = 0
        for path in paths:
            path = path.resolve()
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                rejected += 1
                continue
            if path not in self.files:
                self.files.append(path)
                self.file_states[path] = "state_waiting"
                added += 1
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
                self._set_compatibility_cell(row, None)
        if added and self.table.rowCount() == added:
            self.table.selectRow(0)
        if rejected:
            self._set_status("status_queue_rejected", count=len(self.files), rejected=rejected)
        else:
            self._set_status("status_queue", count=len(self.files))
        if added:
            self._start_preflight()

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
            self.compatibility_results.pop(path, None)
            self.table.removeRow(row)
        if not self.files:
            self._invalidate_preflight()
        self._update_compatibility_panel()
        self._set_status("status_queue", count=len(self.files)) if self.files else self._set_status("status_drop")

    def _clear_files(self) -> None:
        if self._worker_active():
            return
        self._invalidate_preflight()
        self.files.clear()
        self.file_states.clear()
        self.compatibility_results.clear()
        self.table.setRowCount(0)
        self._update_compatibility_panel()
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
        if self._preflight_active() or any(path not in self.compatibility_results for path in self.files):
            self._start_preflight()
            self._set_status("status_preflight", count=sum(
                path not in self.compatibility_results for path in self.files
            ))
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
                elif kind == "compatibility":
                    source, result, generation = payload  # type: ignore[misc]
                    if generation == self.preflight_generation and source in self.files:
                        self.compatibility_results[source] = result
                        row = self._row_for_path(source)
                        if row is not None:
                            self._set_compatibility_cell(row, result)
                        if self._selected_path() == source:
                            self._update_compatibility_panel()
                elif kind == "preflight_done":
                    generation = int(payload)
                    self.preflight_thread = None
                    if generation != self.preflight_generation:
                        self._start_preflight()
                        continue
                    pending = [path for path in self.files if path not in self.compatibility_results]
                    if pending:
                        self._start_preflight()
                    elif not self._worker_active():
                        self.start_button.setEnabled(True)
                        if self.files:
                            self._set_status("status_queue", count=len(self.files))
                        else:
                            self._set_status("status_drop")
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
        self._invalidate_preflight()
        event.accept()


def main() -> int:
    if QApplication is None:
        raise SystemExit("Missing dependency: PySide6. Run setup.bat first.")
    enable_high_dpi_awareness()
    if os.name == "nt":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "zoott28354.ComfyUIConvRotConverter"
            )
        except (AttributeError, OSError):
            pass
    app = QApplication(sys.argv)
    app.setApplicationName("ComfyUI ConvRot Converter")
    app.setStyle("Fusion")
    if APP_ICON.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON)))
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
