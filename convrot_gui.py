from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import ctypes
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import COPY, DND_FILES, TkinterDnD
except ImportError:  # allows helper functions to be imported before setup
    COPY = "copy"
    DND_FILES = None
    TkinterDnD = None


APP_DIR = Path(__file__).resolve().parent
QUANTIZER = APP_DIR / "quant_int8_convrot.py"
SUPPORTED_EXTENSIONS = {".safetensors", ".pth", ".pt", ".ckpt", ".bin"}


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


def output_name(source: Path) -> str:
    """Apply the same output naming rule used by Comfy's official script."""
    converted = re.sub(r"(?i)(bf16|fp16|fp32)", "int8_convrot", source.stem)
    if converted == source.stem:
        converted += "_int8_convrot"
    return converted + ".safetensors"


def output_path(source: Path, output_dir: Path | None) -> Path:
    return (output_dir or source.parent) / output_name(source)


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


class ConvRotApp:
    BG = "#111827"
    PANEL = "#1f2937"
    PANEL_2 = "#243044"
    TEXT = "#f3f4f6"
    MUTED = "#9ca3af"
    ACCENT = "#38bdf8"
    SUCCESS = "#34d399"
    DANGER = "#fb7185"

    def __init__(self) -> None:
        if TkinterDnD is None:
            raise SystemExit("Dipendenza mancante: tkinterdnd2. Esegui prima setup.bat.")
        enable_high_dpi_awareness()
        self.root = TkinterDnD.Tk()
        self.root.title("ComfyUI ConvRot Converter")
        dpi = float(self.root.winfo_fpixels("1i"))
        self.ui_scale = max(1.0, dpi / 96.0)
        self.root.tk.call("tk", "scaling", dpi / 72.0)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = min(round(1000 * self.ui_scale), round(screen_w * 0.88))
        window_h = min(round(820 * self.ui_scale), round(screen_h * 0.86))
        pos_x = max(0, (screen_w - window_w) // 2)
        pos_y = max(0, (screen_h - window_h) // 2)
        self.root.geometry(f"{window_w}x{window_h}+{pos_x}+{pos_y}")
        self.root.minsize(
            min(round(780 * self.ui_scale), round(screen_w * 0.75)),
            min(round(650 * self.ui_scale), round(screen_h * 0.75)),
        )
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.files: list[Path] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False

        self.output_dir = tk.StringVar()
        self.dry_run = tk.BooleanVar(value=False)
        self.mseclip = tk.BooleanVar(value=False)
        self.downcast = tk.BooleanVar(value=False)
        self.write_report = tk.BooleanVar(value=True)
        self.min_gemm = tk.StringVar(value="256")
        self.status = tk.StringVar(value="Trascina qui uno o più modelli")
        self.drop_default_text = "↓  TRASCINA QUI I MODELLI  ↓\n\n.safetensors  .pth  .pt  .ckpt  .bin"

        self._configure_style()
        self._build_ui()
        self.root.after(100, self._poll_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22), foreground=self.TEXT)
        style.configure("Muted.TLabel", foreground=self.MUTED)
        style.configure("TCheckbutton", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", self.PANEL)])
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=(14, 8))
        style.configure("Accent.TButton", background=self.ACCENT, foreground="#082f49")
        style.map("Accent.TButton", background=[("active", "#7dd3fc"), ("disabled", "#374151")])
        style.configure("Danger.TButton", background=self.DANGER, foreground="#4c0519")
        style.configure("Treeview", background=self.PANEL_2, fieldbackground=self.PANEL_2,
                        foreground=self.TEXT, rowheight=28, borderwidth=0)
        style.configure("Treeview.Heading", background=self.PANEL, foreground=self.TEXT,
                        font=("Segoe UI Semibold", 10))
        style.map("Treeview", background=[("selected", "#075985")])

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, padding=24)
        shell.pack(fill="both", expand=True)

        ttk.Label(shell, text="ConvRot Converter", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            shell,
            text="Quantizzazione INT8 + ConvRot per modelli ComfyUI, basata sullo script ufficiale Comfy-Org.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 16))

        self.drop = tk.Label(
            shell,
            text=self.drop_default_text,
            bg=self.PANEL,
            fg=self.ACCENT,
            font=("Segoe UI Semibold", 13),
            relief="flat",
            height=6,
            cursor="hand2",
        )
        self.drop.pack(fill="x")
        self.drop.bind("<Button-1>", lambda _e: self._browse_files())

        buttons = ttk.Frame(shell)
        buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="Aggiungi file", command=self._browse_files).pack(side="left")
        ttk.Button(buttons, text="Rimuovi selezionati", command=self._remove_selected).pack(side="left", padx=8)
        ttk.Button(buttons, text="Svuota", command=self._clear_files).pack(side="left")

        table_frame = ttk.Frame(shell, style="Panel.TFrame")
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=("file", "size", "state"), show="headings", height=6)
        self.table.heading("file", text="Modello")
        self.table.heading("size", text="Dimensione")
        self.table.heading("state", text="Stato")
        self.table.column("file", width=round(560 * self.ui_scale), anchor="w")
        self.table.column("size", width=round(100 * self.ui_scale), anchor="e")
        self.table.column("state", width=round(120 * self.ui_scale), anchor="center")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        options = ttk.Frame(shell, style="Panel.TFrame", padding=14)
        options.pack(fill="x", pady=12)
        ttk.Label(options, text="Cartella output (vuota = accanto all'originale):",
                  background=self.PANEL).grid(row=0, column=0, columnspan=3, sticky="w")
        output_entry = tk.Entry(options, textvariable=self.output_dir, bg=self.PANEL_2, fg=self.TEXT,
                                insertbackground=self.TEXT, relief="flat", font=("Segoe UI", 10))
        output_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 10), ipady=6)
        ttk.Button(options, text="Sfoglia", command=self._browse_output).grid(row=1, column=2, padx=(8, 0), pady=(5, 10))
        ttk.Checkbutton(options, text="Solo analisi (non scrive file)", variable=self.dry_run).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(options, text="MSE clip (sperimentale)", variable=self.mseclip).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(options, text="Riduci FP32 residui", variable=self.downcast).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(options, text="Salva report qualità .tsv", variable=self.write_report).grid(row=3, column=1, sticky="w")
        ttk.Label(options, text="Min GEMM:", background=self.PANEL).grid(row=2, column=2, sticky="e")
        tk.Spinbox(options, from_=0, to=8192, increment=16, textvariable=self.min_gemm, width=8,
                   bg=self.PANEL_2, fg=self.TEXT, buttonbackground=self.PANEL_2, relief="flat").grid(row=3, column=2, sticky="e")
        options.columnconfigure(0, weight=1)
        options.columnconfigure(1, weight=1)

        action = ttk.Frame(shell)
        action.pack(fill="x")
        self.cancel_button = ttk.Button(action, text="Annulla", style="Danger.TButton", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="right", padx=(8, 0))
        self.start_button = ttk.Button(action, text="Avvia conversione", style="Accent.TButton", command=self._start)
        self.start_button.pack(side="right")
        # Pack fixed-size actions before the expanding status label. Otherwise
        # the label can consume the whole row at high Windows scaling factors.
        ttk.Label(action, textvariable=self.status, style="Muted.TLabel").pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )

        log_frame = ttk.Frame(shell, style="Panel.TFrame")
        log_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.log = tk.Text(log_frame, height=9, bg="#0b1220", fg="#d1d5db", insertbackground=self.TEXT,
                           relief="flat", font=("Cascadia Mono", 9), wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        # The widget directly under the mouse receives Windows drop messages.
        # Register every large surface, not only the banner, so dropping anywhere
        # in the window behaves consistently across Tk/Windows versions.
        self._register_drop_targets(
            self.root, shell, self.drop, buttons, table_frame, self.table,
            options, action, log_frame, self.log,
        )

    def _register_drop_targets(self, *widgets: tk.Misc) -> None:
        for widget in widgets:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
            widget.dnd_bind("<<DropLeave>>", self._on_drop_leave)
            widget.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop_enter(self, event: tk.Event) -> str:
        if not (self.worker and self.worker.is_alive()):
            self.drop.configure(bg="#075985", fg="#e0f2fe", text="RILASCIA PER AGGIUNGERE I MODELLI")
            self.status.set("Rilascia i file nella finestra")
        return getattr(event, "action", COPY) or COPY

    def _on_drop_leave(self, event: tk.Event) -> str:
        self._reset_drop_banner()
        return getattr(event, "action", COPY) or COPY

    def _reset_drop_banner(self) -> None:
        self.drop.configure(bg=self.PANEL, fg=self.ACCENT, text=self.drop_default_text)

    @staticmethod
    def _size_label(path: Path) -> str:
        size = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return ""

    def _on_drop(self, event: tk.Event) -> str:
        self._reset_drop_banner()
        if self.worker and self.worker.is_alive():
            self.status.set("Attendi la fine della conversione prima di aggiungere altri file")
            return getattr(event, "action", COPY) or COPY
        try:
            dropped = [Path(item) for item in self.root.tk.splitlist(event.data)]
        except (tk.TclError, TypeError) as exc:
            self.status.set(f"Impossibile leggere i file trascinati: {exc}")
            return getattr(event, "action", COPY) or COPY
        self._add_files(dropped)
        return getattr(event, "action", COPY) or COPY

    def _browse_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Seleziona modelli ComfyUI",
            filetypes=[("Modelli", "*.safetensors *.pth *.pt *.ckpt *.bin"), ("Tutti i file", "*.*")],
        )
        self._add_files([Path(p) for p in paths])

    def _add_files(self, paths: list[Path]) -> None:
        rejected = 0
        for path in paths:
            path = path.resolve()
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                rejected += 1
                continue
            if path not in self.files:
                self.files.append(path)
                self.table.insert("", "end", iid=str(path), values=(path.name, self._size_label(path), "In attesa"))
        self.status.set(f"{len(self.files)} modello/i in coda" + (f" — {rejected} ignorati" if rejected else ""))

    def _remove_selected(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        for iid in self.table.selection():
            path = Path(iid)
            if path in self.files:
                self.files.remove(path)
            self.table.delete(iid)
        self.status.set(f"{len(self.files)} modello/i in coda")

    def _clear_files(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.files.clear()
        self.table.delete(*self.table.get_children())
        self.status.set("Trascina qui uno o più modelli")

    def _browse_output(self) -> None:
        chosen = filedialog.askdirectory(title="Cartella di destinazione")
        if chosen:
            self.output_dir.set(chosen)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self) -> None:
        if not self.files:
            messagebox.showinfo("Nessun modello", "Aggiungi almeno un modello da convertire.")
            return
        try:
            min_gemm = int(self.min_gemm.get())
            if min_gemm < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Valore non valido", "Min GEMM deve essere un numero intero maggiore o uguale a zero.")
            return

        out_dir = Path(self.output_dir.get()).expanduser().resolve() if self.output_dir.get().strip() else None
        if out_dir and not out_dir.is_dir():
            messagebox.showerror("Cartella non valida", "La cartella di destinazione non esiste.")
            return

        if not self.dry_run.get():
            existing = [output_path(source, out_dir) for source in self.files if output_path(source, out_dir).exists()]
            if existing and not messagebox.askyesno(
                "File già esistenti",
                f"{len(existing)} file di destinazione esistono già e saranno sovrascritti. Continuare?",
            ):
                return

        self.cancel_requested = False
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.drop.configure(cursor="arrow")
        self._append_log("\n" + "=" * 72 + "\nAvvio coda ConvRot\n")
        snapshot = list(self.files)
        options = (out_dir, self.dry_run.get(), min_gemm, self.mseclip.get(), self.downcast.get(), self.write_report.get())
        self.worker = threading.Thread(target=self._run_queue, args=(snapshot, options), daemon=True)
        self.worker.start()

    def _run_queue(self, files: list[Path], options: tuple[Path | None, bool, int, bool, bool, bool]) -> None:
        out_dir, dry_run, min_gemm, mseclip, downcast, write_report = options
        successes = 0
        for index, source in enumerate(files, start=1):
            if self.cancel_requested:
                break
            destination = None if dry_run else output_path(source, out_dir)
            report = None
            if write_report and destination is not None:
                report = destination.with_suffix(".quality.tsv")
            command = build_command(source, destination, dry_run=dry_run, min_gemm=min_gemm,
                                    mseclip=mseclip, downcast_fp32=downcast, report_path=report)
            self.events.put(("state", (source, "In corso")))
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
                self.events.put(("log", f"ERRORE avvio: {exc}\n"))
                return_code = -1
            finally:
                self.process = None
            if self.cancel_requested:
                self.events.put(("state", (source, "Annullato")))
                break
            if return_code == 0:
                successes += 1
                self.events.put(("state", (source, "Completato" if not dry_run else "Analizzato")))
            else:
                self.events.put(("state", (source, "Errore")))
        self.events.put(("done", (successes, len(files), self.cancel_requested)))

    def _cancel(self) -> None:
        self.cancel_requested = True
        self.status.set("Annullamento in corso…")
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
                elif kind == "status":
                    self.status.set(str(payload))
                elif kind == "state":
                    source, state = payload  # type: ignore[misc]
                    if self.table.exists(str(source)):
                        values = list(self.table.item(str(source), "values"))
                        values[2] = state
                        self.table.item(str(source), values=values)
                elif kind == "done":
                    successes, total, cancelled = payload  # type: ignore[misc]
                    self.start_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    self.drop.configure(cursor="hand2")
                    self.status.set("Coda annullata" if cancelled else f"Operazione conclusa: {successes}/{total}")
                    self._append_log("\nCoda annullata.\n" if cancelled else f"\nOperazione conclusa: {successes}/{total}.\n")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Conversione attiva", "Interrompere la conversione e chiudere?"):
                return
            self._cancel()
        self.root.destroy()

    def run(self) -> None:
        if not QUANTIZER.is_file():
            messagebox.showerror("Script mancante", f"File non trovato:\n{QUANTIZER}")
            return
        self.root.mainloop()


if __name__ == "__main__":
    ConvRotApp().run()
