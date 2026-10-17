# syntax=docker/dockerfile:1.7
FROM runpod/worker-comfyui:5.8.6-base-cuda12.8.1

ARG SOURCE_REPOSITORY=https://github.com/owen-tech-ramblings/token-gen-flux2-klein-runpod
LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}"
LABEL org.opencontainers.image.description="Full-precision FLUX 3 Image Klein worker for Token-Gen"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

ENV HF_HUB_DISABLE_TELEMETRY=1
ENV MODEL_ROOT=/comfyui/models
ENV COMFY_MODEL_ROOT=/comfyui/models

# Preserve the RunPod handler/start contract and install the immutable ComfyUI
# release that implements FLUX 3 and its Qwen3.5 prompt upsampler.
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/5cc4b870094c89090db54bd8d12649dcaf89f2ba', '/tmp/comfy.tar.gz')" \
    && echo '9779c4188ff39b9c6e719aceddb6457c5ec605b053825a3310aed8253688251c  /tmp/comfy.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/comfy.tar.gz --strip-components=1 -C /comfyui \
    && rm /tmp/comfy.tar.gz \
    && pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r /comfyui/requirements.txt

COPY download_models.py /download_models.py
COPY configure_native_precision.py /configure_native_precision.py
COPY start_worker.py /start_worker.py
COPY token_gen_flux3 /comfyui/custom_nodes/token_gen_flux3
RUN python /configure_native_precision.py
CMD ["python", "/start_worker.py"]
