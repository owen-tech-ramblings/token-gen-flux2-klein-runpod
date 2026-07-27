# Token-Gen FLUX.2 Klein 9B RunPod Worker

A self-contained, scale-to-zero ComfyUI worker for Token-Gen image generation
and native reference-image editing with FLUX.2 Klein 9B distilled.

The container extends the official
`runpod/worker-comfyui:5.8.6-base-cuda12.8.1` image and contains the three
checksum-verified model files. The CUDA 12.8.1/PyTorch cu128 variant prevents
older RunPod hosts from being selected with a PyTorch build their driver cannot
initialize. Keeping the weights in the image lets RunPod schedule the
Blackwell-only endpoint globally instead of pinning it to the data centre of a
network volume.

## Models

- `flux-2-klein-9b-fp8.safetensors`
- `qwen_3_8b_fp8mixed.safetensors`
- `full_encoder_small_decoder.safetensors`

The diffusion checkpoint uses the FLUX Non-Commercial License. The repository
owner must accept Black Forest Labs' agreement and provide a read-only Hugging
Face token as the encrypted GitHub Actions secret `HF_TOKEN`. BuildKit exposes
it only to the gated-model download step. The token is not placed in the
repository, build arguments, image configuration, image layers, or RunPod
endpoint environment.

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
- Container disk: 50 GB

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
