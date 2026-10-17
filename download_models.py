#!/usr/bin/env python3
"""Download and verify pinned FLUX 3 Image Klein model files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class SafeRedirect(HTTPRedirectHandler):
    """Keep gated-model authorization on its HTTPS origin only."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        source = urlsplit(req.full_url)
        destination = urlsplit(newurl)
        if destination.scheme.lower() != "https":
            raise HTTPError(req.full_url, code, "model redirects require HTTPS", headers, fp)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        source_origin = (source.scheme.lower(), source.hostname, source.port if source.port is not None else 443)
        destination_origin = (destination.scheme.lower(), destination.hostname, destination.port if destination.port is not None else 443)
        if redirected is not None and source_origin != destination_origin:
            for name in list(redirected.headers) + list(redirected.unredirected_hdrs):
                if name.lower() == "authorization":
                    redirected.remove_header(name)
        return redirected


# Keep this module boundary patchable by offline downloader tests.
urlopen = build_opener(SafeRedirect()).open


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/runpod-volume/models"))
COMFY_MODEL_ROOT = Path(os.environ.get("COMFY_MODEL_ROOT", "/comfyui/models"))
CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 1024 * 1024 * 1024

MODELS = ({'relative_path': 'diffusion_models/flux3-image-klein-bf16.safetensors', 'url': 'https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/ef6f1128660e522b96e359edcb800b1224d9a0d0/diffusion_models/flux3-image-klein-bf16.safetensors', 'size': 11097404512, 'sha256': 'c3e9813cdd049c831c836b2e9db61150e49a93fc1a4c63b256494642fee0bd7c', 'gated': True}, {'relative_path': 'text_encoders/qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors', 'url': 'https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/ef6f1128660e522b96e359edcb800b1224d9a0d0/text_encoders/qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors', 'size': 18819722392, 'sha256': 'f5062a1fba3e63afe5c88349a311903fec06c959b362f1247127e63d5c860ed4', 'gated': True}, {'relative_path': 'text_encoders/qwen3vl_4b_bf16.safetensors', 'url': 'https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/ef6f1128660e522b96e359edcb800b1224d9a0d0/text_encoders/qwen3vl_4b_bf16.safetensors', 'size': 8875719408, 'sha256': 'f0b24fc5c8d7c0ea6ca909da14c20af5cac53f22506be942c6c956f838b9630f', 'gated': True}, {'relative_path': 'vae/flux2-vae.safetensors', 'url': 'https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/ef6f1128660e522b96e359edcb800b1224d9a0d0/vae/flux2-vae.safetensors', 'size': 336211292, 'sha256': '868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3', 'gated': True})


def selected_models(relative_path: str) -> tuple[dict[str, object], ...]:
    requested = str(relative_path or "").strip()
    if not requested:
        return MODELS
    matches = tuple(
        model for model in MODELS if str(model["relative_path"]) == requested
    )
    if not matches:
        raise RuntimeError(f"unknown model path: {requested}")
    return matches


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
        # A matching marker describes a prior verification, not current bytes.
        # Stream the existing file before allowing any worker startup reuse.
        if sha256_file(target) == expected:
            marker.write_text(expected + "\n", encoding="ascii")
            print(f"model verified: {relative_path}", flush=True)
            return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.is_file() and partial.stat().st_size >= expected_size:
        if partial.stat().st_size == expected_size and sha256_file(partial) == expected:
            os.replace(partial, target)
            marker.write_text(expected + "\n", encoding="ascii")
            print(f"model verified: {relative_path}", flush=True)
            return
        partial.unlink()
    if bool(model["gated"]) and not token:
        raise RuntimeError(
            f"HF_TOKEN is required to download the gated model: {relative_path}"
        )
    base_headers = {"User-Agent": "token-gen-flux3-bootstrap/1.0"}
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

    if destination.resolve(strict=False) == source:
        print(f"model baked: {relative_path}", flush=True)
        return

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
            f"model root parent is unavailable: {MODEL_ROOT.parent}"
        )
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "").strip()
    for model in selected_models(os.environ.get("MODEL_RELATIVE_PATH", "")):
        download_model(model, token)
        link_model(model)


if __name__ == "__main__":
    main()
