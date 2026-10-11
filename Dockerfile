# syntax=docker/dockerfile:1.7
FROM runpod/worker-comfyui:5.8.6-base-cuda12.8.1

ARG SOURCE_REPOSITORY=https://github.com/owen-tech-ramblings/token-gen-flux2-klein-runpod

LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}"
LABEL org.opencontainers.image.description="Self-contained FLUX.2 Klein worker for Token-Gen"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

ENV HF_HUB_DISABLE_TELEMETRY=1
ENV MODEL_ROOT=/comfyui/models
ENV COMFY_MODEL_ROOT=/comfyui/models

COPY download_models.py /download_models.py
COPY configure_native_precision.py /configure_native_precision.py
COPY start_worker.py /start_worker.py

RUN python /configure_native_precision.py

CMD ["python", "/start_worker.py"]
