#!/usr/bin/env python3

import argparse
import asyncio
import sys
from pathlib import Path


COMFYUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMFYUI_DIR))


def parse_script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unet-loader",
        choices=["native", "feather"],
        default="feather",
    )
    parser.add_argument("--distill-lora", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if "--use-flash-attention" not in sys.argv:
        sys.argv.append("--use-flash-attention")
    return args


SCRIPT_ARGS = parse_script_args()


async def load_required_extra_nodes() -> None:
    from nodes import load_custom_node

    await load_custom_node(
        str(COMFYUI_DIR / "comfy_extras" / "nodes_sd3.py"),
        module_parent="comfy_extras",
    )
    await load_custom_node(
        str(COMFYUI_DIR / "custom_nodes" / "ComfyUI-FeatherOps"),
        module_parent="custom_nodes",
    )


def setup_comfyui() -> None:
    from main import apply_custom_paths

    apply_custom_paths()
    asyncio.run(load_required_extra_nodes())


def load_unet(unet_loader: str):
    from nodes import NODE_CLASS_MAPPINGS

    unet_name = "qwen_image_2512_bf16.safetensors"
    if unet_loader == "native":
        return NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
            unet_name=unet_name,
            weight_dtype="default",
        )[0]
    else:
        return NODE_CLASS_MAPPINGS["FeatherUNetLoader"]().load_unet(
            unet_name=unet_name,
            model_type="qwen",
        )[0]


def main() -> None:
    from nodes import NODE_CLASS_MAPPINGS

    cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
    vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    emptylatent = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
    loraloader = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

    model = load_unet(SCRIPT_ARGS.unet_loader)

    if SCRIPT_ARGS.distill_lora:
        model = loraloader.load_lora_model_only(
            model=model,
            lora_name="qwen_image/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
            strength_model=1,
        )[0]
        steps = 4
        cfg = 1
    else:
        steps = 20
        cfg = 5

    clip = cliploader.load_clip(
        clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
        type="qwen_image",
        device="default",
    )[0]

    vae = vaeloader.load_vae(
        vae_name="qwen_image_vae.safetensors",
    )[0]

    positive = cliptextencode.encode(
        text="Anime style, a girl riding a blue horse",
        clip=clip,
    )[0]

    negative = cliptextencode.encode(
        text="",
        clip=clip,
    )[0]

    latent = emptylatent.generate(
        width=1024,
        height=1024,
        batch_size=1,
    )[0]

    latent_sampled = ksampler.sample(
        seed=19260817,
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="simple",
        denoise=1,
        model=model,
        positive=positive,
        negative=negative,
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
