"""Full-precision FLUX 3 prompt expansion with reference and layout fidelity."""
import json
import math
from pathlib import Path
from . import bbox

PROMPTS = json.loads(Path(__file__).with_name('pup_prompts.json').read_text())


def prepare_reference(image, mask, count):
    """Official PUP pixel budgets; keep the diffusion references untouched."""
    import numpy as np
    import torch
    from PIL import Image
    rgb = Image.fromarray((image[0].detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8))
    if mask is not None and tuple(mask.shape[-2:]) == (rgb.height, rgb.width):
        alpha = Image.fromarray(((1 - mask[0].detach().cpu().numpy()).clip(0, 1) * 255).round().astype(np.uint8))
        rgb = Image.composite(rgb, Image.new('RGB', rgb.size, 'white'), alpha)
    width, height = rgb.size
    if max(width, height) / min(width, height) > 200:
        raise ValueError('Prompt expansion reference aspect ratio exceeds 200:1')
    cap = 1024**2 if count == 1 else 724**2 if count == 2 else 512**2
    floor = lambda value: max(32, math.floor(value / 32) * 32)
    w, h = max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)
    if w * h > cap:
        scale = math.sqrt(width * height / cap)
        w, h = floor(width / scale), floor(height / scale)
    if w * h > cap:
        if w == 32:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    rgb = rgb.resize((w, h), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(rgb).copy()).float().div_(255).unsqueeze(0)


class TokenGenFlux3Caption:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'prompt': ('STRING', {'multiline': True}),
                             'boxes_json': ('STRING', {'default': '[]'}),
                             'expand': ('BOOLEAN', {'default': True}),
                             'width': ('INT', {'default': 4096, 'min':256, 'max':4096}),
                             'height': ('INT', {'default': 4096, 'min':256, 'max':4096})},
                'optional': {'clip': ('CLIP',), **{f'image{i}': ('IMAGE',) for i in range(10)},
                             **{f'mask{i}': ('MASK',) for i in range(10)}}}

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'caption'
    CATEGORY = 'Token Gen/FLUX 3'

    def caption(self, prompt, boxes_json, expand, width, height, clip=None, **kwargs):
        images = [kwargs[f'image{i}'] for i in range(10) if kwargs.get(f'image{i}') is not None]
        rows = json.loads(boxes_json)
        if not expand:
            return (f'{prompt.rstrip()} {json.dumps(rows)}' if rows else prompt,)
        if clip is None:
            raise ValueError('FLUX 3 prompt expansion requires its prompt-upscaler checkpoint')
        images = [prepare_reference(kwargs[f'image{i}'], kwargs.get(f'mask{i}'), len(images))
                  for i in range(10) if kwargs.get(f'image{i}') is not None]
        # Detect takes native 0–1000 coordinates. The official plan/reconcile
        # helpers preserve user coordinates even if the language model rewrites them.
        request = f'{prompt.rstrip()} {json.dumps(rows)}' if rows else prompt
        key, request, locked = bbox.plan(request, None, len(images))
        if images:
            request = ('Reference images follow in attachment order: ' +
                       ', '.join(f'ref_image_{i}' for i in range(len(images))) +
                       '\nUser editing instruction: ' + request + '\nOutput the caption.')
        else:
            request = f'TARGET CANVAS: {width}:{height}\n' + request
        tokens = clip.tokenize(request, images=images, min_length=1, thinking=False,
                               skip_template=False, system_prompt=PROMPTS[key])
        result = clip.generate(tokens, do_sample=False, max_length=3584,
                               repetition_penalty=1.0, presence_penalty=0.0 if images else 1.5, mtp=False)
        caption = clip.decode(result).strip()
        if not caption:
            raise RuntimeError('FLUX 3 prompt expansion returned an empty caption')
        if '</think>' in caption:
            caption = caption.split('</think>',1)[1].strip()
        if not caption:
            raise RuntimeError('FLUX 3 prompt expansion returned an empty caption')
        return (bbox.reconcile(caption, locked) if locked else caption,)


NODE_CLASS_MAPPINGS = {'TokenGenFlux3Caption': TokenGenFlux3Caption}
NODE_DISPLAY_NAME_MAPPINGS = {'TokenGenFlux3Caption': 'FLUX 3 Native Prompt Expansion'}
