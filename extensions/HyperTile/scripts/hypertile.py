#   updated for Forge2, based on HyperTile for Forge, based on ...
# https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py 
#Taken from: https://github.com/tfernd/HyperTile/

import gradio as gr
from modules import scripts
from modules.ui_components import InputAccordion
import math
from einops import rearrange
# Use torch rng for consistency across generations
from torch import randint

def random_divisor(value: int, min_value: int, /, max_options: int = 1) -> int:
    min_value = min(min_value, value)

    # All big divisors of value (inclusive)
    divisors = [i for i in range(min_value, value + 1) if value % i == 0]

    ns = [value // i for i in divisors[:max_options]]  # has at least 1 element

    if len(ns) - 1 > 0:
        idx = randint(low=0, high=len(ns) - 1, size=(1,)).item()
    else:
        idx = 0

    return ns[idx]

class HyperTile:
    def patch(self, model, tile_size, swap_size, max_depth, scale_depth):
        latent_tile_size = max(32, tile_size) // 8
        self.temp = None

        def hypertile_in(q, k, v, extra_options):
            model_chans = q.shape[-2]
            shape = extra_options['original_shape']
            apply_to = []
            for i in range(max_depth + 1):
                apply_to.append((shape[-2] / (2 ** i)) * (shape[-1] / (2 ** i)))

            if model_chans in apply_to:
                aspect_ratio = shape[-1] / shape[-2]

                hw = q.size(1)
                h, w = round(math.sqrt(hw * aspect_ratio)), round(math.sqrt(hw / aspect_ratio))

                factor = (2 ** apply_to.index(model_chans)) if scale_depth else 1
                nh = random_divisor(h, latent_tile_size * factor, swap_size)
                nw = random_divisor(w, latent_tile_size * factor, swap_size)

                if nh * nw > 1:
                    q = rearrange(q, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)
                    self.temp = (nh, nw, h, w)
                return q, k, v

            return q, k, v
        def hypertile_out(out, extra_options):
            if self.temp is not None:
                nh, nw, h, w = self.temp
                self.temp = None
                out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)
            return out

        m = model.clone()
        m.set_model_attn1_patch(hypertile_in)
        m.set_model_attn1_output_patch(hypertile_out)
        return (m, )


class HyperTileForForge(scripts.Script):
    sorting_priority = 13.5

    def title(self):
        return "HyperTile"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            tile_size = gr.Slider(label='Tile Size', minimum=32, maximum=2048, step=1, value=256)
            swap_size = gr.Slider(label='Swap Size', minimum=1, maximum=128, step=1, value=2)
            max_depth = gr.Slider(label='Max Depth', minimum=0, maximum=10, step=1, value=2)
            scale_depth = gr.Checkbox(label='Scale Depth', value=False)

        self.infotext_fields = [
            (enabled, lambda d: d.get("HyperTile_enabled", False)),
            (tile_size,     "HyperTile_tile_size"),
            (swap_size,     "HyperTile_swap_size"),
            (max_depth,     "HyperTile_max_depth"),
            (scale_depth,   "HyperTile_scale_depth"),
        ]

        return enabled, tile_size, swap_size, max_depth, scale_depth

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, tile_size, swap_size, max_depth, scale_depth = script_args
        tile_size, swap_size, max_depth = int(tile_size), int(swap_size), int(max_depth)

        if not enabled:
            return

#        if not shared.sd_model.is_webui_legacy_model():     #   ideally would be is_flux
#            gr.Info ("HyperTile is not compatible with Flux")
#            return

        unet = p.sd_model.forge_objects.unet

        unet = HyperTile().patch(unet, tile_size, swap_size, max_depth, scale_depth)[0]

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            HyperTile_enabled       = enabled,
            HyperTile_tile_size     = tile_size,
            HyperTile_swap_size     = swap_size,
            HyperTile_max_depth     = max_depth,
            HyperTile_scale_depth   = scale_depth,
        ))

        return