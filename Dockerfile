# syntax=docker/dockerfile:1.7

FROM runpod/worker-comfyui:5.8.6-base

ARG SOURCE_REPOSITORY=https://github.com/owen-tech-ramblings/token-gen-flux2-klein-runpod

LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}"
LABEL org.opencontainers.image.description="Private FLUX.2 Klein 9B ComfyUI worker for Token-Gen"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only AND LicenseRef-FLUX-Non-Commercial"

ENV HF_HUB_DISABLE_TELEMETRY=1

# The diffusion checkpoint is gated by Black Forest Labs. BuildKit mounts the
# Hugging Face token only for this layer; the credential is never copied into
# the image or recorded as an ARG/ENV value.
RUN --mount=type=secret,id=hf_token,required=true \
    test -s /run/secrets/hf_token \
    && wget --quiet --tries=5 --timeout=60 \
        --header="Authorization: Bearer $(tr -d '\r\n' < /run/secrets/hf_token)" \
        --output-document=/comfyui/models/diffusion_models/flux-2-klein-9b-fp8.safetensors \
        "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/902d9d510b51533e07729f19211414a3648b77d2/flux-2-klein-9b-fp8.safetensors" \
    && echo "865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee  /comfyui/models/diffusion_models/flux-2-klein-9b-fp8.safetensors" | sha256sum --check -

RUN wget --quiet --tries=5 --timeout=60 \
        --output-document=/comfyui/models/text_encoders/qwen_3_8b_fp8mixed.safetensors \
        "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/23fbc8aa8b621f29f2249cd1bd9c47e5d0eebd83/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" \
    && echo "abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6  /comfyui/models/text_encoders/qwen_3_8b_fp8mixed.safetensors" | sha256sum --check -

RUN wget --quiet --tries=5 --timeout=60 \
        --output-document=/comfyui/models/vae/full_encoder_small_decoder.safetensors \
        "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/a3efc24f613ef42d9428af62fdbd6f5fd8856c4a/full_encoder_small_decoder.safetensors" \
    && echo "ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62  /comfyui/models/vae/full_encoder_small_decoder.safetensors" | sha256sum --check -

