#!/usr/bin/env python3

import asyncio
import sys
from pathlib import Path


COMFYUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMFYUI_DIR))

if "--use-flash-attention" not in sys.argv:
    sys.argv.append("--use-flash-attention")


def setup_comfyui() -> None:
    from main import apply_custom_paths
    from nodes import load_custom_node

    apply_custom_paths()
    asyncio.run(
        load_custom_node(
            str(COMFYUI_DIR / "comfy_extras" / "nodes_edit_model.py"),
            module_parent="comfy_extras",
        )
    )


def main() -> None:
    from nodes import NODE_CLASS_MAPPINGS

    unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
    cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
    vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    referencelatent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
    conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

    model = unetloader.load_unet(
        unet_name="flux-2-klein-9b-bf16.safetensors",
        weight_dtype="default",
    )[0]

    clip = cliploader.load_clip(
        clip_name="qwen_3_8b_fp8mixed.safetensors",
        type="flux2",
        device="default",
    )[0]

    vae = vaeloader.load_vae(
        vae_name="flux2_vae.safetensors",
    )[0]

    image = loadimage.load_image(
        image="ComfyUI_00001_.png",
    )[0]

    latent = vaeencode.encode(
        vae=vae,
        pixels=image,
    )[0]

    condition_positive = cliptextencode.encode(
        text="Show the full body view of this character.",
        clip=clip,
    )[0]

    condition_positive_reference = referencelatent.execute(
        conditioning=condition_positive,
        latent=latent,
    )[0]

    condition_negative = conditioningzeroout.zero_out(
        conditioning=condition_positive,
    )[0]

    latent_sampled = ksampler.sample(
        seed=19260817,
        steps=4,
        cfg=1,
        sampler_name="euler",
        scheduler="simple",
        denoise=1,
        model=model,
        positive=condition_positive_reference,
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
