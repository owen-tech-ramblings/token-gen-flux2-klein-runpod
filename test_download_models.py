import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request

import download_models


class DownloadModelsTests(unittest.TestCase):
    def test_default_bootstrap_and_startup_use_native_precision(self):
        # Changing the default back to FP8, the small decoder, or automatic
        # startup casting must fail at the consuming bootstrap/start boundary.
        fixtures = {
            "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/ef6f1128660e522b96e359edcb800b1224d9a0d0/text_encoders/qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors": b"native-upsampler",
            "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/ef6f1128660e522b96e359edcb800b1224d9a0d0/diffusion_models/flux3-image-klein-bf16.safetensors": b"native-diffusion",
            "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/ef6f1128660e522b96e359edcb800b1224d9a0d0/text_encoders/qwen3vl_4b_bf16.safetensors": b"native-encoder",
            "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/ef6f1128660e522b96e359edcb800b1224d9a0d0/vae/flux2-vae.safetensors": b"full-native-vae",
        }

        def fixture_urlopen(request, timeout):
            self.assertEqual(timeout, 120)
            self.assertIn(request.full_url, fixtures)
            return io.BytesIO(fixtures[request.full_url])

        fixture_models = tuple(
            {**model, "sha256": hashlib.sha256(fixtures.get(str(model["url"]), b"wrong-model")).hexdigest(),
             "size": len(fixtures.get(str(model["url"]), b"wrong-model"))}
            for model in download_models.MODELS
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertIsNotNone(importlib.util.find_spec("start_worker"), "native files must be verified before ComfyUI starts")
            import start_worker

            expected = {
                "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors": b"native-upsampler",
                "diffusion_models/flux3-image-klein-bf16.safetensors": b"native-diffusion",
                "text_encoders/qwen3vl_4b_bf16.safetensors": b"native-encoder",
                "vae/flux2-vae.safetensors": b"full-native-vae",
            }
            launched = []

            def exec_verified_start(path, args):
                self.assertEqual(path, "/start.sh")
                self.assertEqual(args, ["/start.sh"])
                self.assertEqual(
                    {str(path.relative_to(root / "models")): path.read_bytes()
                     for path in (root / "models").rglob("*.safetensors")}, expected,
                )
                self.assertNotIn("HF_TOKEN", os.environ)
                launched.append(True)

            with (
                patch.object(download_models, "MODEL_ROOT", root / "models"),
                patch.object(download_models, "COMFY_MODEL_ROOT", root / "models"),
                patch.object(download_models, "MODELS", fixture_models),
                patch.object(download_models, "urlopen", fixture_urlopen),
                patch.dict(os.environ, {"HF_TOKEN": "fixture", "MODEL_RELATIVE_PATH": ""}),
                patch.object(start_worker.os, "execv", exec_verified_start),
            ):
                start_worker.main()
            self.assertEqual(launched, [True])

            cached_diffusion = root / "models/diffusion_models/flux3-image-klein-bf16.safetensors"
            intact = expected["diffusion_models/flux3-image-klein-bf16.safetensors"]
            cached_diffusion.write_bytes(intact[:-1] + b"!")
            marker = cached_diffusion.with_name(cached_diffusion.name + ".sha256")
            self.assertEqual(marker.read_text().strip(), hashlib.sha256(intact).hexdigest())
            self.assertEqual(cached_diffusion.stat().st_size, len(intact))
            refused_launches = []
            with (
                patch.object(download_models, "MODEL_ROOT", root / "models"),
                patch.object(download_models, "COMFY_MODEL_ROOT", root / "models"),
                patch.object(download_models, "MODELS", fixture_models),
                patch.object(download_models, "urlopen", fixture_urlopen),
                patch.dict(os.environ, {"HF_TOKEN": "", "MODEL_RELATIVE_PATH": ""}),
                patch.object(start_worker.os, "execv", lambda *_args: refused_launches.append(True)),
            ):
                with self.assertRaisesRegex(RuntimeError, "HF_TOKEN is required"):
                    start_worker.main()
            self.assertEqual(refused_launches, [])
            with (
                patch.object(download_models, "MODEL_ROOT", root / "models"),
                patch.object(download_models, "COMFY_MODEL_ROOT", root / "models"),
                patch.object(download_models, "MODELS", fixture_models),
                patch.object(download_models, "urlopen", fixture_urlopen),
                patch.dict(os.environ, {"HF_TOKEN": "fixture", "MODEL_RELATIVE_PATH": ""}),
                patch.object(start_worker.os, "execv", exec_verified_start),
            ):
                start_worker.main()
            self.assertEqual(launched, [True, True])

            for failure in ("missing_token", "denied_access", "wrong_checksum", "partial_selection"):
                with self.subTest(failure=failure):
                    models = fixture_models
                    token = "" if failure == "missing_token" else "fixture"
                    if failure == "wrong_checksum":
                        models = ({**models[0], "sha256": "0" * 64}, *models[1:])

                    def failure_urlopen(request, timeout):
                        if failure == "denied_access":
                            raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)
                        return fixture_urlopen(request, timeout)

                    with (
                        patch.object(download_models, "MODEL_ROOT", root / failure),
                        patch.object(download_models, "COMFY_MODEL_ROOT", root / failure),
                        patch.object(download_models, "MODELS", models),
                        patch.object(download_models, "urlopen", failure_urlopen),
                        patch.object(download_models.time, "sleep", lambda _seconds: None),
                        patch.dict(os.environ, {"HF_TOKEN": token, "MODEL_RELATIVE_PATH": str(models[1]["relative_path"]) if failure == "partial_selection" else ""}),
                        patch.object(start_worker.os, "execv", lambda *_args: self.fail("unverified worker launched")),
                    ):
                        with self.assertRaises(RuntimeError):
                            start_worker.main()

            import configure_native_precision

            start = root / "start.sh"
            start.write_text(
                '#!/bin/bash\nif [[ ${SERVE_API_LOCALLY:-false} == true ]]; then\n'
                'python -u /comfyui/main.py --listen &\nwait\npython -u /handler.py --rp_serve_api\nelse\n'
                'python -u /comfyui/main.py --disable-auto-launch &\nwait\npython -u /handler.py\nfi\n',
                encoding="utf-8",
            )
            configure_native_precision.configure_start(start)
            configure_native_precision.configure_start(start)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            fake_python.write_text(
                '#!/usr/bin/python3\nimport json, os, sys\n'
                'with open(os.environ["PRECISION_ARGS_FILE"] + (".handler" if sys.argv[2] == "/native_handler.py" else ""), "w") as output:\n'
                '    json.dump(sys.argv[1:], output)\n', encoding="utf-8",
            )
            fake_python.chmod(0o755)
            for local in ("false", "true"):
                output = root / (local + ".json")
                environment = {**os.environ, "PATH": str(fake_bin),
                               "SERVE_API_LOCALLY": local, "PRECISION_ARGS_FILE": str(output)}
                subprocess.run(["/bin/bash", str(start)], env=environment, check=True)
                args = json.loads(output.read_text())
                self.assertEqual(args[:5], ["-u", "/comfyui/main.py", "--bf16-unet", "--bf16-text-enc", "--fp32-vae"])
                self.assertEqual(args.count("--bf16-unet"), 1)
                handler_args = json.loads(output.with_name(output.name + ".handler").read_text())
                self.assertEqual(handler_args[:2], ["-u", "/native_handler.py"])
            original = start.read_text()
            for broken in ("no native startup", original.replace("--fp32-vae", "--bf16-vae")):
                start.write_text(broken)
                with self.assertRaises(RuntimeError):
                    configure_native_precision.configure_start(start)
                self.assertEqual(start.read_text(), broken)

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

        for intact in (True, False):
            with self.subTest(full_partial_intact=intact), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                model = {
                    "relative_path": "models/complete.bin",
                    "url": "https://example.invalid/complete.bin",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                    "gated": intact,
                }
                target = root / "models/complete.bin"
                partial = target.with_name(target.name + ".partial")
                partial.parent.mkdir(parents=True)
                partial.write_bytes(payload if intact else payload[:-1] + b"!")

                def complete_urlopen(request, timeout):
                    if intact:
                        self.fail("a complete verified partial must publish without networking")
                    self.assertIsNone(request.get_header("Range"), "a complete corrupt partial must restart from byte zero")
                    return io.BytesIO(payload)

                with (
                    patch.object(download_models, "MODEL_ROOT", root),
                    patch.object(download_models, "urlopen", complete_urlopen),
                ):
                    download_models.download_model(model, "")
                self.assertEqual(target.read_bytes(), payload)
                self.assertFalse(partial.exists())
                self.assertEqual(target.with_name(target.name + ".sha256").read_text().strip(), model["sha256"])

    def test_gated_download_requires_token_when_model_is_missing(self):
        opener = getattr(download_models.urlopen, "__self__", None)
        handler = next((item for item in getattr(opener, "handlers", ())
                        if isinstance(item, HTTPRedirectHandler)), HTTPRedirectHandler())
        request = Request("https://huggingface.co/model", headers={
            "Authorization": "Bearer fixture", "Range": "bytes=42-",
        })
        for destination, expected_token in (
            ("https://huggingface.co/renamed-model", "Bearer fixture"),
            ("https://HUGGINGFACE.CO:443/renamed-model", "Bearer fixture"),
            ("https://signed-cdn.example.invalid/model?signature=fixture", None),
            ("https://huggingface.co:8443/model", None),
        ):
            with self.subTest(destination=destination):
                redirected = handler.redirect_request(request, None, 302, "Found", {}, destination)
                self.assertEqual(redirected.get_header("Authorization"), expected_token)
                self.assertEqual(redirected.get_header("Range"), "bytes=42-")
        for destination in ("http://huggingface.co/model", "ftp://example.invalid/model"):
            with self.subTest(insecure_redirect=destination):
                with self.assertRaises(HTTPError):
                    handler.redirect_request(request, None, 302, "Found", {}, destination)

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
