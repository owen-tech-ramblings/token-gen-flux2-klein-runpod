# Token-Gen FLUX.2 Klein 9B RunPod Worker

Private, scale-to-zero ComfyUI worker for Token-Gen image generation and native
reference-image editing with FLUX.2 Klein 9B distilled.

The worker extends the official `runpod/worker-comfyui:5.8.6-base` image and
bakes these pinned, checksum-verified model files into the container:

- `flux-2-klein-9b-fp8.safetensors`
- `qwen_3_8b_fp8mixed.safetensors`
- `full_encoder_small_decoder.safetensors`

The diffusion checkpoint is covered by the FLUX Non-Commercial License and
requires the Hugging Face account building this image to accept Black Forest
Labs' agreement. This repository and its GHCR package must remain private.

## Build

1. Accept the agreement on the
   [FLUX.2 Klein 9B FP8 model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8).
2. Create a read-only Hugging Face token for that account.
3. Store it as the private repository Actions secret `HF_TOKEN`.
4. Dispatch `.github/workflows/publish.yml`.

The token is passed as a BuildKit secret. It is not an image layer, build
argument, environment variable, repository file, or workflow output.

## RunPod configuration

- Image: immutable `ghcr.io/owen-tech-ramblings/token-gen-flux2-klein-runpod:sha-...` tag
- Registry credential: private GHCR read credential
- GPU pool: start with 24 GB and validate; use a larger pool only if the 9B
  model cannot complete the canary suite without offloading or OOM
- GPUs per worker: 1
- Minimum workers: 0
- Maximum workers: 1
- Scaling: queue delay
- Idle timeout: 180 seconds
- Execution timeout: 600 seconds
- FlashBoot: enabled
- Container disk: at least 50 GB
- Network volume: none

## Token-Gen API settings

```text
RUNPOD_IMAGE_MODEL_FAMILY=flux2_klein
FLUX_DIFFUSION_MODEL=flux-2-klein-9b-fp8.safetensors
FLUX_TEXT_ENCODER=qwen_3_8b_fp8mixed.safetensors
FLUX_CLIP_MODEL=
FLUX_VAE_MODEL=full_encoder_small_decoder.safetensors
FLUX_IMAGE_ID=flux2-klein-9b
FLUX_IMAGE_NAME=FLUX.2 Klein 9B
FLUX_DEFAULT_STEPS=4
FLUX_GUIDANCE=1.0
```

Do not cut over the live endpoint until the worker passes generation,
reference-edit, masked-edit, exact-text, and rollback canaries.

## Acceptance canaries

The promotion gate is based on actual output, not job completion alone:

- Exact text: poster containing `FAMILY MOVIE NIGHT` with every character
  readable and correctly ordered.
- Dense text: café menu with three short item names and prices.
- Generation: photorealistic scene with hands, faces, and small objects.
- Native edit: add a helmet to an existing rider while preserving identity,
  motorcycle, pose, framing, and background.
- Text edit: replace only the wording on an existing sign.
- Masked edit: change only the selected region and byte-visually preserve the
  unmasked scene.

The existing FLUX.1 Schnell image reference and endpoint configuration must be
recorded before promotion so rollback does not require rebuilding anything.
