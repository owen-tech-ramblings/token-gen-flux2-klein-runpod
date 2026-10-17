# Token Gen FLUX 3 Image Klein RunPod worker

The existing image endpoint serves FLUX 3 Image Klein generation and editing with
native BF16 generator, Qwen3-VL text encoder and Qwen3.5-9B prompt expansion, plus
full F32 VAE execution. The repository/container name is retained for continuity.

`download_models.py` pins the four official model files by revision, byte size and
SHA256. The bootstrap downloads and verifies them before launching ComfyUI and
removes HF_TOKEN from the inference process environment. The Hugging Face account
must have access to `black-forest-labs/flux-3-image-klein-comfy`.

ComfyUI is pinned to `5cc4b870094c89090db54bd8d12649dcaf89f2ba`; PyTorch 2.11 uses CUDA
12.8 wheels. The existing single BLACKWELL_96 GPU, 200 GB ephemeral disk,
scale-to-zero worker configuration and RunPod workflow/image response contract
remain. No alternate endpoint or model fallback is created.

`TokenGenFlux3Caption` uses the released BF16 prompt upsampler with the official
BFL task prompts and layout reconciliation. Every reference is supplied in order
without cropping or forcing it to another reference's aspect ratio. Caller boxes
remain bound to their original coordinates. Caption failure fails the job.

The API owns quality, canvas, seed, count, reference and mask controls. The native
checkpoint recipe is 16 steps, Euler/simple, CFG 1.0. High-quality output uses
lossless PNG. Lower step counts or canvas sizes require explicit caller choices.
The distilled checkpoint is BFL's recommended generation/editing model; the base
checkpoint is intended for fine-tuning and research.

Build with the existing `publish.yml` workflow. Deploy its immutable digest to the
existing Token Gen image endpoint together with the API's versioned FLUX 3 config.
Focused checks: `python3 -m unittest test_flux3_caption test_download_models.DownloadModelsTests.test_default_bootstrap_and_startup_use_native_precision`.

Upstream documentation: https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy
