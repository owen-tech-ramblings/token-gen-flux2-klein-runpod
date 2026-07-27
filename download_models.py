#!/usr/bin/env python3
"""Populate a RunPod network volume with pinned FLUX.2 Klein model files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/runpod-volume/models"))
COMFY_MODEL_ROOT = Path(os.environ.get("COMFY_MODEL_ROOT", "/comfyui/models"))
CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 1024 * 1024 * 1024

MODELS = (
    {
        "relative_path": "diffusion_models/flux-2-klein-9b-fp8.safetensors",
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/902d9d510b51533e07729f19211414a3648b77d2/flux-2-klein-9b-fp8.safetensors",
        "sha256": "865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee",
        "size": 9_433_061_528,
        "gated": True,
    },
    {
        "relative_path": "text_encoders/qwen_3_8b_fp8mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/23fbc8aa8b621f29f2249cd1bd9c47e5d0eebd83/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
        "sha256": "abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6",
        "size": 8_664_848_742,
        "gated": False,
    },
    {
        "relative_path": "vae/full_encoder_small_decoder.safetensors",
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/a3efc24f613ef42d9428af62fdbd6f5fd8856c4a/full_encoder_small_decoder.safetensors",
        "sha256": "ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62",
        "size": 249_519_092,
        "gated": False,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state(path: Path) -> hashlib._Hash:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest


def download_model(model: dict[str, object], token: str) -> None:
    relative_path = str(model["relative_path"])
    target = MODEL_ROOT / relative_path
    expected = str(model["sha256"])
    expected_size = int(model["size"])
    marker = target.with_name(target.name + ".sha256")

    if target.is_file() and target.stat().st_size == expected_size:
        if marker.is_file() and marker.read_text(encoding="ascii").strip() == expected:
            print(f"model ready: {relative_path}", flush=True)
            return
        if sha256_file(target) == expected:
            marker.write_text(expected + "\n", encoding="ascii")
            print(f"model verified: {relative_path}", flush=True)
            return

    if bool(model["gated"]) and not token:
        raise RuntimeError(
            f"HF_TOKEN is required to download the gated model: {relative_path}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.is_file() and partial.stat().st_size > expected_size:
        partial.unlink()
    base_headers = {"User-Agent": "token-gen-flux2-bootstrap/1.0"}
    if bool(model["gated"]):
        base_headers["Authorization"] = f"Bearer {token}"

    last_error: Exception | None = None
    for attempt in range(1, 4):
        downloaded = partial.stat().st_size if partial.is_file() else 0
        digest = sha256_state(partial) if downloaded else hashlib.sha256()
        next_progress = ((downloaded // PROGRESS_BYTES) + 1) * PROGRESS_BYTES
        headers = dict(base_headers)
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        try:
            action = "resuming" if downloaded else "downloading"
            print(
                f"{action} model: {relative_path} from {downloaded // (1024 * 1024)} MiB "
                f"(attempt {attempt}/3)",
                flush=True,
            )
            request = Request(str(model["url"]), headers=headers)
            with urlopen(request, timeout=120) as response:
                response_status = getattr(response, "status", None)
                append = bool(downloaded and response_status == 206)
                if downloaded and not append:
                    print(
                        f"server did not honor resume for {relative_path}; restarting file",
                        flush=True,
                    )
                    downloaded = 0
                    digest = hashlib.sha256()
                    next_progress = PROGRESS_BYTES
                with partial.open("ab" if append else "wb") as output:
                    while chunk := response.read(CHUNK_BYTES):
                        output.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_progress:
                            print(
                                f"downloaded {downloaded // (1024 * 1024)} MiB: {relative_path}",
                                flush=True,
                            )
                            next_progress += PROGRESS_BYTES
                    output.flush()
                    os.fsync(output.fileno())
            actual = digest.hexdigest()
            if actual != expected:
                partial.unlink(missing_ok=True)
                raise RuntimeError(
                    f"checksum mismatch for {relative_path}: expected {expected}, got {actual}"
                )
            if partial.stat().st_size != expected_size:
                raise RuntimeError(
                    f"size mismatch for {relative_path}: expected {expected_size}, got {partial.stat().st_size}"
                )
            os.replace(partial, target)
            marker.write_text(expected + "\n", encoding="ascii")
            print(f"model verified: {relative_path}", flush=True)
            return
        except (HTTPError, URLError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(5 * attempt)

    raise RuntimeError(f"failed to download {relative_path}") from last_error


def link_model(model: dict[str, object]) -> None:
    """Expose a volume-backed model through ComfyUI's native model tree."""
    relative_path = Path(str(model["relative_path"]))
    source = (MODEL_ROOT / relative_path).resolve(strict=True)
    destination = COMFY_MODEL_ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink() and destination.resolve() == source:
        print(f"model linked: {relative_path}", flush=True)
        return

    temporary = destination.with_name(destination.name + ".link")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source)
    os.replace(temporary, destination)
    print(f"model linked: {relative_path}", flush=True)


def main() -> None:
    if not MODEL_ROOT.parent.is_dir():
        raise RuntimeError(
            f"RunPod network volume is not mounted at {MODEL_ROOT.parent}"
        )
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "").strip()
    for model in MODELS:
        download_model(model, token)
        link_model(model)


if __name__ == "__main__":
    main()
