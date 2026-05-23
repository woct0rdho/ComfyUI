#!/usr/bin/env python3

import sys
from pathlib import Path


COMFYUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMFYUI_DIR))

if "--use-flash-attention" not in sys.argv:
    sys.argv.append("--use-flash-attention")


def setup_comfyui() -> None:
    from main import apply_custom_paths

    apply_custom_paths()


def main() -> None:
    from nodes import NODE_CLASS_MAPPINGS

    checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

    model, clip, vae = checkpointloadersimple.load_checkpoint(
        ckpt_name="sdxl_anime/WAI-Illustrious-v16.safetensors",
    )

    condition_positive = cliptextencode.encode(
        text="1girl, masterpiece, best quality",
        clip=clip,
    )[0]

    condition_negative = cliptextencode.encode(
        text="worst quality, low quality",
        clip=clip,
    )[0]

    latent = emptylatentimage.generate(
        width=768,
        height=1152,
        batch_size=1,
    )[0]

    latent_sampled = ksampler.sample(
        seed=19260817,
        steps=20,
        cfg=5,
        sampler_name="euler",
        scheduler="simple",
        denoise=1,
        model=model,
        positive=condition_positive,
        negative=condition_negative,
        latent_image=latent,
    )[0]

    output = vaedecode.decode(
        samples=latent_sampled,
        vae=vae,
    )[0]

    saveimage.save_images(
        filename_prefix="ComfyUI",
        images=output,
    )


if __name__ == "__main__":
    setup_comfyui()

    import torch

    with torch.inference_mode():
        main()
