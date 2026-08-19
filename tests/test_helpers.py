# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 zoott28354 and contributors

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The tested helpers do not execute tensor operations. Lightweight stubs keep this
# suite runnable on CI and development machines without downloading CUDA PyTorch.
try:
    import torch  # noqa: F401
except ImportError:
    torch_stub = types.ModuleType("torch")

    class Tensor:  # minimal isinstance target used by the checkpoint reader
        pass

    torch_stub.Tensor = Tensor
    torch_stub.float16 = object()
    torch_stub.bfloat16 = object()
    torch_stub.float32 = object()
    torch_stub.float64 = object()
    torch_stub.int8 = object()
    torch_stub.uint8 = object()
    torch_stub.linspace = lambda *args, **kwargs: []
    torch_stub.no_grad = lambda: (lambda function: function)
    sys.modules["torch"] = torch_stub

try:
    import safetensors  # noqa: F401
except ImportError:
    safetensors_stub = types.ModuleType("safetensors")
    safetensors_stub.safe_open = lambda *args, **kwargs: None
    safetensors_torch_stub = types.ModuleType("safetensors.torch")
    safetensors_torch_stub.save_file = lambda *args, **kwargs: None
    sys.modules["safetensors"] = safetensors_stub
    sys.modules["safetensors.torch"] = safetensors_torch_stub

try:
    import comfy_kitchen  # noqa: F401
except ImportError:
    comfy_stub = types.ModuleType("comfy_kitchen")
    comfy_tensor_stub = types.ModuleType("comfy_kitchen.tensor")
    comfy_int8_stub = types.ModuleType("comfy_kitchen.tensor.int8")
    comfy_int8_stub._build_hadamard = lambda *args, **kwargs: None
    comfy_int8_stub._rotate_weight = lambda *args, **kwargs: None
    sys.modules["comfy_kitchen"] = comfy_stub
    sys.modules["comfy_kitchen.tensor"] = comfy_tensor_stub
    sys.modules["comfy_kitchen.tensor.int8"] = comfy_int8_stub

from convrot_gui import (
    TRANSLATIONS,
    build_command,
    numbered_output_path,
    output_name,
    output_path,
    plan_output_paths,
)
from quant_int8_convrot import (
    atomic_save_model,
    atomic_write_quality_report,
    classify_decoder_text,
    classify_umt5_text,
    detect_text_encoder_preset,
    open_model,
    paths_refer_to_same_file,
)


class HelperTests(unittest.TestCase):
    def test_setup_auto_detects_python_without_selection_menu(self):
        setup = (PROJECT_ROOT / "setup.bat").read_text(encoding="utf-8")
        self.assertNotIn("Select the Python installation", setup)
        self.assertNotIn("Choice [1-5]", setup)
        self.assertIn("for %%V in (3.14 3.13 3.12)", setup)
        self.assertIn("Compatible Python was not found.", setup)

    def test_translations_have_identical_keys(self):
        self.assertEqual(set(TRANSLATIONS["it"]), set(TRANSLATIONS["en"]))
        self.assertEqual(TRANSLATIONS["en"]["start"], "Start conversion")

    def test_output_name_replaces_precision(self):
        self.assertEqual(output_name(Path("wan_FP16_v2.safetensors")), "wan_int8_convrot_v2.safetensors")

    def test_output_name_appends_suffix(self):
        self.assertEqual(output_name(Path("wan.safetensors")), "wan_int8_convrot.safetensors")

    def test_output_path_uses_selected_directory(self):
        self.assertEqual(
            output_path(Path("C:/models/a_bf16.safetensors"), Path("D:/converted")),
            Path("D:/converted/a_int8_convrot.safetensors"),
        )

    def test_build_dry_run_has_no_destination(self):
        command = build_command(
            Path("model.safetensors"), None, dry_run=True, min_gemm=256,
            mseclip=False, downcast_fp32=False, report_path=None,
        )
        self.assertIn("--dry-run", command)
        self.assertEqual(command[-2:], ["--min-gemm", "256"])

    def test_build_conversion_options(self):
        command = build_command(
            Path("model.safetensors"), Path("out.safetensors"), dry_run=False, min_gemm=0,
            mseclip=True, downcast_fp32=True, report_path=Path("report.tsv"),
        )
        self.assertIn("--mseclip", command)
        self.assertIn("--downcast-fp32", command)
        self.assertIn("--verify-report", command)

    def test_numbered_output_name(self):
        self.assertEqual(
            numbered_output_path(Path("model.safetensors"), 2),
            Path("model (2).safetensors"),
        )

    def test_duplicate_names_receive_numbers_in_shared_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "converted"
            output.mkdir()
            first = root / "folder-a" / "model.safetensors"
            second = root / "folder-b" / "model.safetensors"
            planned = plan_output_paths([first, second], output, True)
            self.assertEqual(planned[first], output / "model_int8_convrot.safetensors")
            self.assertEqual(planned[second], output / "model_int8_convrot (1).safetensors")

    def test_existing_model_or_report_advances_number(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "model.safetensors"
            (root / "model_int8_convrot.safetensors").write_text("existing", encoding="utf-8")
            planned = plan_output_paths([source], None, True)
            self.assertEqual(planned[source], root / "model_int8_convrot (1).safetensors")

            (root / "model_int8_convrot.safetensors").unlink()
            (root / "model_int8_convrot.quality.tsv").write_text("existing", encoding="utf-8")
            planned = plan_output_paths([source], None, True)
            self.assertEqual(planned[source], root / "model_int8_convrot (1).safetensors")

    def test_output_never_replaces_another_queued_source(self):
        first = Path("models/model_bf16.safetensors")
        second = Path("models/model_int8_convrot.safetensors")
        planned = plan_output_paths([first, second], None, False)
        self.assertEqual(planned[first], Path("models/model_int8_convrot (1).safetensors"))

    def test_pytorch_checkpoint_opens_without_extra_flag(self):
        with mock.patch("quant_int8_convrot._TorchReader") as reader:
            open_model("model.ckpt")
        reader.assert_called_once_with("model.ckpt")

    def test_path_identity_and_atomic_quality_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "quality.tsv"
            self.assertTrue(paths_refer_to_same_file(report, report.parent / "." / report.name))
            atomic_write_quality_report(report, [(1.25, 0.999, 256, "layer\tname")])
            self.assertEqual(
                report.read_text(encoding="utf-8"),
                "relerr_pct\tcosine\tgroupsize\tlayer\n1.2500\t0.999000\t256\tlayer\\tname\n",
            )
            self.assertEqual(list(Path(folder).glob("*.partial")), [])

    def test_atomic_model_save_replaces_only_after_success(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "model.safetensors"
            destination.write_text("old", encoding="utf-8")

            def successful_save(tensors, path, metadata):
                Path(path).write_text("new", encoding="utf-8")

            with mock.patch("quant_int8_convrot.save_file", side_effect=successful_save):
                atomic_save_model({}, destination, {})

            self.assertEqual(destination.read_text(encoding="utf-8"), "new")
            self.assertEqual(list(Path(folder).glob("*.partial")), [])

    def test_atomic_model_save_preserves_existing_file_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "model.safetensors"
            destination.write_text("old", encoding="utf-8")

            def failed_save(tensors, path, metadata):
                Path(path).write_text("incomplete", encoding="utf-8")
                raise OSError("simulated write failure")

            with mock.patch("quant_int8_convrot.save_file", side_effect=failed_save):
                with self.assertRaises(OSError):
                    atomic_save_model({}, destination, {})

            self.assertEqual(destination.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(Path(folder).glob("*.partial")), [])


class TextEncoderPresetTests(unittest.TestCase):
    def test_detects_standalone_umt5(self):
        keys = ["shared.weight"]
        for block in range(24):
            keys.extend([
                f"encoder.block.{block}.layer.0.SelfAttention.q.weight",
                f"encoder.block.{block}.layer.1.DenseReluDense.wi_0.weight",
            ])
        self.assertEqual(detect_text_encoder_preset(keys), "umt5_text")

    def test_detects_comfy_native_umt5(self):
        keys = ["token_embedding.weight", "norm.weight"]
        for block in range(24):
            keys.extend([
                f"blocks.{block}.attn.q.weight",
                f"blocks.{block}.ffn.fc1.weight",
                f"blocks.{block}.pos_embedding.embedding.weight",
            ])
        self.assertEqual(detect_text_encoder_preset(keys), "umt5_text")

    def test_detects_qwen_and_protects_edges_and_visual_tower(self):
        keys = ["model.embed_tokens.weight"]
        for block in range(4):
            keys.append(f"model.layers.{block}.self_attn.q_proj.weight")
        keys.append("visual.blocks.0.attn.qkv.weight")
        self.assertEqual(detect_text_encoder_preset(keys), "qwen_text")
        self.assertFalse(classify_decoder_text(keys[1], (4096, 4096), 3, "qwen")[0])
        self.assertTrue(classify_decoder_text(keys[2], (4096, 4096), 3, "qwen")[0])
        self.assertFalse(classify_decoder_text(keys[-1], (4096, 4096), 3, "qwen")[0])

    def test_detects_gemma_markers(self):
        keys = ["model.embed_tokens.weight", "vision_tower.embeddings.weight"]
        keys.extend(f"model.layers.{block}.mlp.up_proj.weight" for block in range(4))
        self.assertEqual(detect_text_encoder_preset(keys), "gemma_text")
        self.assertFalse(classify_decoder_text(
            "vision_model.encoder.layers.1.self_attn.q_proj", (4096, 4096), 3, "gemma"
        )[0])

    def test_filename_disambiguates_text_only_gemma(self):
        keys = ["model.embed_tokens.weight"]
        keys.extend(f"model.layers.{block}.mlp.up_proj.weight" for block in range(4))
        self.assertEqual(detect_text_encoder_preset(keys, "gemma_2_2b_bf16.safetensors"), "gemma_text")

    def test_detects_qwen_vl_language_model_prefix(self):
        keys = ["model.language_model.embed_tokens.weight"]
        keys.extend(
            f"model.language_model.layers.{block}.self_attn.q_proj.weight" for block in range(4)
        )
        keys.append("model.visual.blocks.0.attn.qkv.weight")
        self.assertEqual(detect_text_encoder_preset(keys, "qwen3vl_4b_bf16.safetensors"), "qwen_text")

    def test_aio_bundle_does_not_get_text_encoder_preset(self):
        keys = ["shared.weight", "model.diffusion_model.blocks.0.self_attn.q.weight"]
        keys.extend(
            f"text_encoders.umt5.transformer.encoder.block.{block}.layer.0.SelfAttention.q.weight"
            for block in range(4)
        )
        self.assertIsNone(detect_text_encoder_preset(keys))

    def test_umt5_classifier_keeps_shared_embedding(self):
        self.assertTrue(classify_umt5_text(
            "encoder.block.3.layer.0.SelfAttention.q", (4096, 4096)
        )[0])
        self.assertFalse(classify_umt5_text("shared", (250112, 4096))[0])


if __name__ == "__main__":
    unittest.main()
