FROM runpod/worker-comfyui:5.8.6-base

ARG SOURCE_REPOSITORY=https://github.com/owen-tech-ramblings/token-gen-flux2-klein-runpod

LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}"
LABEL org.opencontainers.image.description="FLUX.2 Klein network-volume bootstrap worker for Token-Gen"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

ENV HF_HUB_DISABLE_TELEMETRY=1

COPY download_models.py /download_models.py
COPY bootstrap-start.sh /bootstrap-start.sh
RUN chmod 0755 /bootstrap-start.sh

CMD ["/bootstrap-start.sh"]
