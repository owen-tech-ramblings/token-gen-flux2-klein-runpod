# Token-Gen FLUX.2 Klein 9B RunPod Worker

A lightweight, scale-to-zero ComfyUI bootstrap worker for Token-Gen image
generation and native reference-image editing with FLUX.2 Klein 9B distilled.

The container extends the official
`runpod/worker-comfyui:5.8.6-base-cuda12.8.1` image but contains no model
weights or credentials. The CUDA 12.8.1/PyTorch cu128 variant prevents older
RunPod hosts from being selected with a PyTorch build their driver cannot
initialize. On startup it verifies the three pinned model files on the attached
RunPod network volume. Missing files are downloaded once, checksum-verified,
and retained across scale-to-zero cycles.

## Models

- `flux-2-klein-9b-fp8.safetensors`
- `qwen_3_8b_fp8mixed.safetensors`
- `full_encoder_small_decoder.safetensors`

The diffusion checkpoint uses the FLUX Non-Commercial License. The RunPod
endpoint owner must accept Black Forest Labs' agreement and provide a read-only
Hugging Face token as the private `HF_TOKEN` endpoint environment variable for
the initial download. The token is not placed in this repository or container.

After the volume has been populated successfully, the endpoint can be updated
to remove `HF_TOKEN`; subsequent workers verify and use the existing files.
At startup, the worker links the verified volume files into ComfyUI's native
`diffusion_models`, `text_encoders`, and `vae` directories so every loader sees
the same persistent files without copying them into the container disk.

## RunPod configuration

- Image: immutable
  `ghcr.io/owen-tech-ramblings/token-gen-flux2-klein-runpod:sha-...` tag
- Network volume: at least 25 GB, mounted by RunPod at `/runpod-volume`
- GPU: prefer 32 GB for the first 9B canary; test 24 GB only after successful
  32 GB inference
- GPUs per worker: 1
- Minimum workers: 0
- Maximum workers: 1
- Scaling: queue delay
- Idle timeout: 180 seconds
- Execution timeout: 900 seconds for the initial download canary
- FlashBoot: enabled
- Container disk: 30 GB

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
