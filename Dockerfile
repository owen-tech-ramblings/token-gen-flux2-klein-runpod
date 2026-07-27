# syntax=docker/dockerfile:1.7
FROM runpod/worker-comfyui:5.8.6-base-cuda12.8.1

ARG SOURCE_REPOSITORY=https://github.com/owen-tech-ramblings/token-gen-flux2-klein-runpod

LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}"
LABEL org.opencontainers.image.description="Self-contained FLUX.2 Klein worker for Token-Gen"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

ENV HF_HUB_DISABLE_TELEMETRY=1

COPY download_models.py /download_models.py

RUN --mount=type=secret,id=hf_token,required=true \
    token="$(tr -d '\r\n' < /run/secrets/hf_token)" && \
    token="${token#HF_TOKEN=}" && \
    test -n "${token}" && \
    MODEL_ROOT=/comfyui/models \
    COMFY_MODEL_ROOT=/comfyui/models \
    MODEL_RELATIVE_PATH=diffusion_models/flux-2-klein-9b-fp8.safetensors \
    HF_TOKEN="${token}" \
    python /download_models.py

RUN MODEL_ROOT=/comfyui/models \
    COMFY_MODEL_ROOT=/comfyui/models \
    MODEL_RELATIVE_PATH=text_encoders/qwen_3_8b_fp8mixed.safetensors \
    python /download_models.py

RUN MODEL_ROOT=/comfyui/models \
    COMFY_MODEL_ROOT=/comfyui/models \
    MODEL_RELATIVE_PATH=vae/full_encoder_small_decoder.safetensors \
    python /download_models.py

CMD ["/start.sh"]
