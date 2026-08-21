# Changelog

## 1.0.0 — 2026-08-21

First public release of ComfyUI ConvRot Converter for Windows.

### Features

- Modern PySide6 interface with native drag and drop.
- Automatic model compatibility analysis before conversion.
- Sequential batch processing for models stored in different folders.
- INT8 + ConvRot conversion using `comfy-kitchen`.
- Detection of the actual FP16, BF16, FP8, or INT8 source storage dtype.
- Protected presets for LTX-2.3, UMT5, Gemma, and Qwen text encoders.
- Optional experimental MSE clipping and matching output names.
- Per-layer TSV quality reports.
- Automatic numbered output names without overwriting existing files.
- Optional deletion of the original only after successful conversion.
- English and Italian interface.
- High-DPI support for 4K Windows displays.

### Requirements

- Windows 10 or Windows 11.
- Standard 64-bit Python 3.12, 3.13, or 3.14.
- NVIDIA CUDA-compatible GPU.
- Sufficient RAM, VRAM, and disk space for the selected model.

Run `setup.bat` for the initial installation. It creates `start.bat` automatically.

This project is licensed under GNU GPL-3.0 and is independent from Comfy-Org.
