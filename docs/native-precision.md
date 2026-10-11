# FLUX.2 Klein 9B distilled: native precision

The currently exposed distilled identity retains four sampling steps and
guidance 1.0. This change removes the previous FP8 diffusion/encoder and small
decoder efficiency choices from its default worker model set. It does not swap to
the undistilled Base model, invent extra steps, or add an unadvertised preview
fallback.

Primary recipe: [Black Forest Labs' distilled model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
uses BF16, four steps, and guidance 1.0. Its approximate memory example does not
establish peak consumption for this application's 2048 generation/edit canvas.

The [Comfy Org native files](https://huggingface.co/Comfy-Org/flux2-klein-9B)
package the original Qwen3-8B encoder and full FLUX.2 VAE for the existing
ComfyUI loader shapes. The [ComfyUI Klein guide](https://docs.comfy.org/tutorials/flux/flux-2-klein)
documents the model directories and native workflows.

| File | Immutable revision | Bytes | SHA256 |
| --- | --- | ---: | --- |
| BFL `flux-2-klein-9b.safetensors` | `92196c8e11f7b6cf2b7493e037d8c5345c559216` | 18,157,185,168 | `0975d6b77b5f510b99547d6724a208e36527df654e8f6134f59ece3f9f30da58` |
| Comfy Org `split_files/text_encoders/qwen_3_8b.safetensors` | `3f62d9d8ae1fec33c6e91453d5c712855b096b55` | 16,381,517,176 | `f0ff9239d56269ca1d05e5f86da6a79fac111af464955681f11c7ab0ec5ef6c1` |
| Comfy Org `split_files/vae/flux2-vae.safetensors` | `3f62d9d8ae1fec33c6e91453d5c712855b096b55` | 336,211,292 | `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3` |

Evidence collected 2026-10-11: anonymous Hugging Face metadata reports these
revisions, sizes and LFS hashes. Anonymous HEAD of the pinned diffusion file
returns 401 `GatedRepo`; encoder and VAE return 200. Header-only range reads
of the public files show all 399 encoder tensors BF16 and 250 VAE tensors F32
(plus one integer tensor). No tensor payload was downloaded locally.

The existing Hugging Face credential must have accepted access to the BF16 diffusion
repository, independently of access to its FP8 repository. This was not checked
by reading or fetching credentials. Startup fails if access, size, or hash
verification fails. Comfy Org's repository currently redirects to
`vae-text-encorder-for-flux-klein-9b`; the pinned original URLs resolve and the
checksum remains authoritative.

[RunPod's base startup](https://github.com/runpod-workers/worker-comfyui/blob/5.8.6/src/start.sh)
has separate local-API and serverless ComfyUI invocations. The build applies
`--bf16-unet --bf16-text-enc --fp32-vae` to both. It requires the expected two
invocations and rejects conflicting or partial precision flags. This retains
the base worker's handler, readiness, GPU checks, signal behavior and model
paths. The [ComfyUI precision arguments](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/cli_args.py)
support these native modes. No FP32 diffusion upcast is presented as improved
model quality.

The image contains code/dependencies only. `start_worker.py` runs the existing
verified downloader against `/comfyui/models` on the worker's 200 GB ephemeral
disk. It refuses partial model selection and only executes the original
`/start.sh` after every file passes verification. `HF_TOKEN` is removed before
ComfyUI/handler execution. No network volume or local/hosted build-time model
download is needed. Governed endpoint promotion must configure the existing
credential and `RUNPOD_INIT_TIMEOUT=1800` (or a longer measured adequate value),
as described by [RunPod's initialization guidance](https://docs.runpod.io/serverless/development/optimization).

## API handoff

`integration/api-native-precision.patch` changes only family-specific default
filenames/identity in `server_details_api.py`. The existing Klein `UNETLoader`
already requests `weight_dtype=default`, and `CLIPLoader(type=flux2)` / the full
`VAELoader` accept these native files. No node topology change is required.
`integration/native-api-settings.env.example` records the nonsecret settings
that must replace existing filename overrides during governed promotion. No
runtime file is changed by this workstream.

The previous worker image plus previous filename settings must remain recorded
as one rollback snapshot. The separate API source patch and worker image cannot
be cut over independently while jobs use the old filenames.

## Verification and remaining proof

The five selected checks exercise real bootstrap output files with tiny offline
payloads, checksum/reuse, partial-download resume, missing gated credentials,
model-tree linking, and both shell startup modes with an isolated argv sink.
The native boundary regression first fails against the previous FP8 manifest.
It also rejects unexpected startup shapes, conflicting precision flags,
missing/denied HF access, checksum mismatch, and partial model selection before
the worker launch boundary.
Reuse streams and rehashes actual model bytes even when a previous checksum
marker matches. A complete interrupted `.partial` is hash-verified and promoted
before networking; a complete corrupt partial is discarded and redownloaded
from byte zero. The existing startup/resume selectors cover stale same-size
corruption, refused launch without repair access, verified repair before launch,
and both complete-partial outcomes.
The downloader's installed redirect handler retains authorization only within
the same HTTPS origin, strips it on cross-host or changed-port redirects, and
rejects HTTP/FTP redirects. The existing auth selector checks emitted redirect
requests and the original missing-token refusal; signed HF CDN requests receive
their signed URL without the Hugging Face bearer header.

Local checks do not prove a full container build, authenticated BF16 access,
peak 96 GB memory use, generation/edit/inpaint fidelity, or RunPod cold start.
Those are explicit build/canary outcomes to collect before endpoint promotion.
No paid generation or live endpoint change is authorized in this workstream.
