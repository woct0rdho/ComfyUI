#!/usr/bin/env python3

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


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
        str(COMFYUI_DIR / "comfy_extras" / "nodes_lt.py"),
        module_parent="comfy_extras",
    )
    await load_custom_node(
        str(COMFYUI_DIR / "comfy_extras" / "nodes_video.py"),
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

    unet_name = "ltx-2.3-22b-dev_transformer_only_bf16.safetensors"
    if unet_loader == "native":
        return NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
            unet_name=unet_name,
            weight_dtype="default",
        )[0]
    else:
        return NODE_CLASS_MAPPINGS["FeatherUNetLoader"]().load_unet(
            unet_name=unet_name,
            model_type="default",
        )[0]


def main() -> None:
    from nodes import NODE_CLASS_MAPPINGS

    dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
    vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    emptylatentvideo = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
    ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
    loraloader = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
    savevideo = NODE_CLASS_MAPPINGS["SaveVideo"]()

    model = load_unet(SCRIPT_ARGS.unet_loader)

    if SCRIPT_ARGS.distill_lora:
        model = loraloader.load_lora_model_only(
            model=model,
            lora_name="ltx/ltx-2.3-22b-distilled-lora-v1.1-r384-fro0.9.safetensors",
            strength_model=1,
        )[0]
        steps = 8
        cfg = 1
    else:
        steps = 20
        cfg = 3

    clip = dualcliploader.load_clip(
        clip_name1="gemma_3_12b_it_fp4_mixed.safetensors",
        clip_name2="ltx-2.3_text_projection_bf16.safetensors",
        type="ltxv",
        device="default",
    )[0]

    vae = vaeloader.load_vae(
        vae_name="LTX23_video_vae_bf16.safetensors",
    )[0]

    positive = cliptextencode.encode(
        text="Anime style, a girl riding a blue horse",
        clip=clip,
    )[0]

    negative = cliptextencode.encode(
        text="",
        clip=clip,
    )[0]

    positive, negative = ltxvconditioning.execute(
        positive=positive,
        negative=negative,
        frame_rate=24,
    )

    latent = emptylatentvideo.generate(
        width=768,
        height=768,
        length=57,
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

    images = vaedecode.decode(
        samples=latent_sampled,
        vae=vae,
    )[0]

    video = createvideo.execute(
        images=images,
        fps=24,
        audio=None,
    )[0]

    type(savevideo).hidden = SimpleNamespace(prompt=None, extra_pnginfo=None)
    savevideo.execute(
        video=video,
        filename_prefix="video/ComfyUI",
        format="auto",
        codec="auto",
    )


if __name__ == "__main__":
    setup_comfyui()

    import torch

    with torch.inference_mode():
        main()
