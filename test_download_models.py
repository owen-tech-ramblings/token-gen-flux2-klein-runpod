import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_models


class DownloadModelsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
