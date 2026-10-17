"""Keep the pinned RunPod ComfyUI handler, replacing only its result transport."""
import runpod
from handler import handler
from image_delivery import wrap_handler

if __name__ == '__main__':
    runpod.serverless.start({'handler': wrap_handler(handler)})
