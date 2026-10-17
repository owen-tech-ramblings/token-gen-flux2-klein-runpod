#!/usr/bin/env python3
"""Pin ComfyUI's native precision in both RunPod base startup modes."""

from pathlib import Path


def configure_start(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    prefix = "python -u /comfyui/main.py "
    native = prefix + "--bf16-unet --bf16-text-enc --fp32-vae "
    incompatible = ("--fp8", "--fp4", "--fp16", "--bf16-vae", "--fp32-unet", "--fp32-text-enc")
    if any(flag in text for flag in incompatible) or text.count(prefix) != 2:
        raise RuntimeError("unexpected ComfyUI base startup; native precision was not configured")
    handler = "python -u /handler.py"
    replacement = "python -u /native_handler.py"
    if text.count(handler) == 2:
        text = text.replace(handler, replacement)
    elif text.count(replacement) != 2:
        raise RuntimeError("unexpected RunPod handler startup")
    if text.count(native) == 2:
        path.write_text(text, encoding="utf-8")
        return
    if any(flag in text for flag in ("--bf16-unet", "--bf16-text-enc", "--fp32-vae")):
        raise RuntimeError("partial native precision configuration in ComfyUI base startup")
    path.write_text(text.replace(prefix, native), encoding="utf-8")


if __name__ == "__main__":
    configure_start(Path("/start.sh"))
