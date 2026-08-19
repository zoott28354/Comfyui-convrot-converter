import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convrot_gui import build_command, output_name, output_path
from quant_int8_convrot import (
    classify_decoder_text,
    classify_umt5_text,
    detect_text_encoder_preset,
)


class HelperTests(unittest.TestCase):
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
