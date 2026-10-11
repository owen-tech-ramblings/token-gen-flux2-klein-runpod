# Token-Gen FLUX.2 Klein 9B RunPod Worker

A self-contained, scale-to-zero ComfyUI worker for Token-Gen image generation
and native reference-image editing with FLUX.2 Klein 9B distilled.

The container extends the official
`runpod/worker-comfyui:5.8.6-base-cuda12.8.1` image and downloads all three
checksum-verified native model files before ComfyUI starts. The CUDA 12.8.1/PyTorch cu128 variant prevents
older RunPod hosts from being selected with a PyTorch build their driver cannot
initialize. Models use the worker's ephemeral local disk, keeping the code
image small and allowing global Blackwell scheduling without a network volume.
An empty disk has a longer verified-download cold start; no inference worker
starts after a partial or failed bootstrap.

## Models

- `flux-2-klein-9b.safetensors`: original distilled BF16 diffusion weights
- `qwen_3_8b.safetensors`: native BF16 Qwen3-8B text encoder
- `flux2-vae.safetensors`: full native encoder and decoder, stored/executed F32

All files are pinned by immutable repository revision, byte size, and SHA256 in
`download_models.py`. This image contains no FP8 or small-decoder fallback.
The distilled recipe remains four inference steps and guidance 1.0. Both
ComfyUI startup modes explicitly select BF16 diffusion/text encoding and F32
VAE execution, rejecting an unexpected base startup during the build.

The files total 34,874,913,636 bytes (32.48 GiB), before runtime activations.
The 96 GB Blackwell pool provides substantial headroom; actual peak memory and
output quality at the API's ultra canvas still require the acceptance canaries.
See [native precision evidence](docs/native-precision.md) for sources and the
separate API handoff.

The diffusion checkpoint uses the FLUX Non-Commercial License. The repository
owner must accept Black Forest Labs' BF16 repository agreement and configure
the existing read-only Hugging Face credential as the RunPod secret environment
variable `HF_TOKEN` during governed cutover. It is read only by the startup
downloader and removed before ComfyUI/handler execution. It is not placed in
the repository, build arguments, image configuration, or image layers. Building
the code-only image requires no Hugging Face credential.

## RunPod configuration

- Image: immutable
  `ghcr.io/owen-tech-ramblings/token-gen-flux2-klein-runpod:sha-...` tag
- Network volume: none
- Data centre restriction: none
- GPU pool: `BLACKWELL_96` only
- GPUs per worker: 1
- Minimum workers: 0
- Maximum workers: 1
- Scaling: queue delay
- Idle timeout: 180 seconds
- Execution timeout: 1200 seconds
- FlashBoot: Priority
- Container disk: 200 GB ephemeral local disk
- Initialization timeout: `RUNPOD_INIT_TIMEOUT=1800`, increased if cold-start
  measurement demonstrates that more time is needed

## Token-Gen API settings

```text
RUNPOD_IMAGE_MODEL_FAMILY=flux2_klein
FLUX_DIFFUSION_MODEL=flux-2-klein-9b.safetensors
FLUX_TEXT_ENCODER=qwen_3_8b.safetensors
FLUX_CLIP_MODEL=
FLUX_VAE_MODEL=flux2-vae.safetensors
FLUX_IMAGE_ID=flux2-klein-9b
FLUX_IMAGE_NAME=FLUX.2 Klein 9B
FLUX_DEFAULT_STEPS=4
FLUX_GUIDANCE=1.0
```

Do not cut over the live endpoint until the worker passes generation,
reference-edit, masked-edit, exact-text, and rollback canaries.
The settings above must be promoted together with the reviewed worker image.
Existing machine-local filename overrides take precedence over API defaults;
the separate API patch alone cannot update those overrides.

## Candidate build

The existing manual `publish.yml` workflow builds the code/dependency image
without fetching model files or reading `HF_TOKEN`. Dispatch it on the reviewed `codex/*` branch
to build an immutable full-SHA candidate without updating `latest`.
Only a default-branch dispatch can update `latest`; neither dispatch changes a
RunPod endpoint. The workflow runs five explicitly named checks.

No successful GPU canary or container build is implied by local downloader
checks. Record initialization duration for the full 34.9 GB bootstrap, confirm
the configured deadline, and capture actual 96 GB memory and output quality
before endpoint promotion. No quantized fallback exists to mask a failed cold
start. [RunPod initialization guidance](https://docs.runpod.io/serverless/development/optimization)
documents the initialization deadline setting.

## Acceptance canaries

- Exact text: poster containing `FAMILY MOVIE NIGHT` with every character
  readable and correctly ordered.
- Dense text: café menu with three short item names and prices.
- Generation: photorealistic scene with hands, faces, and small objects.
- Native edit: add a helmet to an existing rider while preserving identity,
  motorcycle, pose, framing, and background.
- Text edit: replace only the wording on an existing sign.
- Masked edit: change only the selected region and preserve the unmasked scene.

The existing FLUX.1 Schnell image reference and endpoint configuration must be
recorded before promotion so rollback does not require rebuilding anything.
