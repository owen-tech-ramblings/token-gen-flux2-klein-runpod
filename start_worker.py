#!/usr/bin/env python3
"""Verify every native model before starting the existing RunPod worker."""

import os

import download_models


def main() -> None:
    if os.environ.get("MODEL_RELATIVE_PATH", "").strip():
        raise RuntimeError("worker startup requires the complete native model set")
    download_models.main()
    # Authentication is only needed by the gated bootstrap, not ComfyUI or its
    # handler. A failed bootstrap never reaches this launch boundary.
    os.environ.pop("HF_TOKEN", None)
    os.execv("/start.sh", ["/start.sh"])


if __name__ == "__main__":
    main()
