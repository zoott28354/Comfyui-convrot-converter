# ComfyUI ConvRot Converter for Windows

A bilingual drag-and-drop interface for converting ComfyUI models to **INT8 + ConvRot**. The conversion is performed by Comfy-Org's official [`quant_int8_convrot.py`](https://github.com/Comfy-Org/comfy-model-tools/blob/main/quant_int8_convrot.py) script, with protected presets for selected model families.

## Installation

1. Install [64-bit Python 3.12](https://www.python.org/downloads/) and make sure **Python Launcher** is selected.
2. Double-click `setup.bat`. It creates a local `.venv` and installs CUDA PyTorch, `comfy-kitchen`, `safetensors`, and drag-and-drop support.
3. For subsequent launches, simply run `START.bat`.

The first setup downloads PyTorch and may require several gigabytes. Conversion requires a CUDA-compatible NVIDIA GPU and enough VRAM. The source model is never modified.

The setup ignores additional package indexes configured globally in `pip` and uses only PyPI and the official PyTorch CUDA index. This prevents unreachable corporate or NVIDIA mirrors from blocking installation.

## Language

The application starts in Italian. Use the **ITA / ENG** selector in the upper-right corner to switch the complete interface to English at runtime. Labels, queue states, dialogs, validation messages, and application-generated log entries are localized. Output from the underlying Comfy-Org conversion script remains in its original language.

## Usage

1. Drop one or more `.safetensors`, `.pth`, `.pt`, `.ckpt`, or `.bin` files anywhere in the application window.
2. Leave the output folder blank to save next to each source, or choose a different destination.
3. For an unfamiliar architecture, run **Analysis only** first. It reports which layers would be converted without writing an output file.
4. Click **Start conversion**.

The **source storage dtypes** section in the log reports the actual precision stored in the model header (`F16`, `BF16`, `F8_E4M3`, or `I8`). Do not rely only on the filename or on `compute/passthrough dtype`, which describes the precision selected for non-quantized layers. The converter warns when the source is already predominantly FP8 or INT8 to avoid double quantization.

## Protected automatic presets

### LTX-2.3

The converter automatically recognizes 48-block LTX-2.3 bundles and applies the protected Comfy-Org selection. It quantizes all 34 Linear layers in blocks 2–45, for a total of 1,496 layers; blocks 0, 1, 46, and 47 and the audio/video connectors remain at source precision. The small `to_gate_logits` layers are included by bypassing **Min GEMM** only for this preset. Conversion stops if the detected structure does not produce exactly 1,496 layers.

### Standalone text encoders

Standalone text encoders are automatically recognized and use conservative presets:

- **UMT5 / UMT5-XXL:** quantizes attention and feed-forward projections in every encoder block while preserving the shared embedding, normalization, and final layers.
- **Gemma:** quantizes only attention and MLP projections in internal language blocks. Embeddings, the vision tower, output heads, and the first and last language blocks remain BF16/FP16.
- **Qwen / Qwen-VL:** applies the same protection to embeddings, visual components, output heads, and the first and last language blocks.

Text-encoder presets validate that every selected block has a uniform structure and stop if the checkpoint does not match the recognized family. AIO checkpoints that also contain a diffusion model are not mistaken for standalone text encoders.

From the command line, selection can be controlled with `--preset auto`, `--preset ltx2_official`, `--preset umt5_text`, `--preset gemma_text`, `--preset qwen_text`, or `--preset generic`. The GUI uses `auto`.

## Interface and file handling

Drag-and-drop works across the entire window, including the queue table and log panel. On Windows, do not run `START.bat` as administrator: Windows blocks dragging from a non-elevated File Explorer process into an elevated application.

The interface enables per-monitor Windows DPI awareness and scales its initial size and table columns automatically, including 4K displays at 150–200% desktop scaling.

Output names follow the official script: `model_bf16.safetensors` becomes `model_int8_convrot.safetensors`. If the original name does not contain `bf16`, `fp16`, or `fp32`, `_int8_convrot` is appended.

## Options

- **Min GEMM 256:** recommended official default. Skips layers that are too small for INT8 to provide a useful performance benefit.
- **MSE clip:** experimental mode that can reduce weight reconstruction error; always validate the result.
- **Downcast remaining FP32:** converts selected non-quantized FP32 layers to the compute dtype to reduce output size.
- **Quality report:** writes a `.quality.tsv` file containing relative error, cosine similarity, and group size for every quantized layer.

## Important notes

- ConvRot is applied only to automatically detected compatible layers. Some architectures or loaders that remap weight keys may not be suitable; use Analysis only first.
- Models are processed sequentially to limit memory usage.
- The application asks for confirmation before overwriting an existing destination file.

The included Comfy-Org script is distributed under GPL-3.0. A copy of its license is provided in `LICENSE-COMFY-MODEL-TOOLS`.
