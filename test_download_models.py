import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_models


class DownloadModelsTests(unittest.TestCase):
    def test_selected_models_can_limit_a_build_layer(self):
        requested = str(download_models.MODELS[1]["relative_path"])
        selected = download_models.selected_models(requested)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["relative_path"], requested)
        with self.assertRaisesRegex(RuntimeError, "unknown model path"):
            download_models.selected_models("missing/model.safetensors")

    def test_download_verifies_and_reuses_marker(self):
        payload = b"token-gen-model-test"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            source.write_bytes(payload)
            model = {
                "relative_path": "models/test.bin",
                "url": source.as_uri(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "gated": False,
            }
            with patch.object(download_models, "MODEL_ROOT", root / "volume"):
                download_models.download_model(model, "")
                target = download_models.MODEL_ROOT / "models/test.bin"
                marker = target.with_name(target.name + ".sha256")
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual(marker.read_text().strip(), model["sha256"])
                model["url"] = "file:///does-not-exist"
                download_models.download_model(model, "")

    def test_download_resumes_an_existing_partial_file(self):
        prefix = b"already-downloaded-"
        suffix = b"remaining-bytes"
        payload = prefix + suffix

        class PartialResponse(io.BytesIO):
            status = 206

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = {
                "relative_path": "models/resume.bin",
                "url": "https://example.invalid/resume.bin",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "gated": False,
            }
            target = root / "models/resume.bin"
            partial = target.with_name(target.name + ".partial")
            partial.parent.mkdir(parents=True)
            partial.write_bytes(prefix)

            def fake_urlopen(request, timeout):
                self.assertEqual(timeout, 120)
                self.assertEqual(request.get_header("Range"), f"bytes={len(prefix)}-")
                return PartialResponse(suffix)

            with (
                patch.object(download_models, "MODEL_ROOT", root),
                patch.object(download_models, "urlopen", fake_urlopen),
            ):
                download_models.download_model(model, "")

            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(partial.exists())

    def test_gated_download_requires_token_when_model_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = {
                "relative_path": "models/gated.bin",
                "url": "file:///does-not-exist",
                "sha256": hashlib.sha256(b"x").hexdigest(),
                "size": 1,
                "gated": True,
            }
            with patch.object(download_models, "MODEL_ROOT", Path(temp_dir)):
                with self.assertRaisesRegex(RuntimeError, "HF_TOKEN is required"):
                    download_models.download_model(model, "")

    def test_link_model_exposes_volume_file_in_comfy_model_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            volume_root = root / "volume"
            comfy_root = root / "comfy"
            relative_path = Path("text_encoders/test.safetensors")
            source = volume_root / relative_path
            source.parent.mkdir(parents=True)
            source.write_bytes(b"model")
            model = {"relative_path": str(relative_path)}

            with (
                patch.object(download_models, "MODEL_ROOT", volume_root),
                patch.object(download_models, "COMFY_MODEL_ROOT", comfy_root),
            ):
                download_models.link_model(model)
                download_models.link_model(model)

            destination = comfy_root / relative_path
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.resolve(), source.resolve())
            self.assertEqual(destination.read_bytes(), b"model")

    def test_link_model_is_a_noop_for_baked_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            relative_path = Path("vae/test.safetensors")
            source = root / relative_path
            source.parent.mkdir(parents=True)
            source.write_bytes(b"model")
            model = {"relative_path": str(relative_path)}

            with (
                patch.object(download_models, "MODEL_ROOT", root),
                patch.object(download_models, "COMFY_MODEL_ROOT", root),
            ):
                download_models.link_model(model)

            self.assertFalse(source.is_symlink())
            self.assertEqual(source.read_bytes(), b"model")


if __name__ == "__main__":
    unittest.main()
