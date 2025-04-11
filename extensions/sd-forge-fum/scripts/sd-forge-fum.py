import torch
import gradio as gr
import random

from modules import scripts
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from modules.ui_components import InputAccordion

def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


def patch_FUM_v2(unet_patcher, b1, b2, s1, s2):
    model_channels = unet_patcher.model.diffusion_model.config.get("model_channels")

    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        process = FUMForForge.doFUM

        if process:
            scale = scale_dict.get(h.shape[1], None)
            if scale is not None:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)

                if hsp.device not in on_cpu_devices:
                    try:
                        hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                    except:
                        print("Device", hsp.device, "does not support the torch.fft functions used in the FUM node, switching to CPU.")
                        on_cpu_devices[hsp.device] = True
                        hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
                else:
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

        return h, hsp

    m = unet_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


class FUMForForge(scripts.Script):
    sorting_priority = 12  # It will be the 12th item on UI.
    
    doFUM = True
    random_move_enabled = False  # Variable to control random UNet movement
    simple_move_s1_enabled = False # Variable to control S1 UNet movement
    
    last_s1 = 0.99  # Store last values of S1 and S2
    last_s2 = 0.95

    presets_builtin = [
        #   name, b1, b2, s1, s2, start step, end step
        ('Forge default', 1.01, 1.02, 0.99, 0.95, 0.0, 1.0),
        ('SD 1.4', 1.3, 1.4, 0.9, 0.2, 0.0, 1.0),
        ('SD 1.5', 1.5, 1.6, 0.9, 0.2, 0.0, 1.0),
        ('SD 2.1', 1.4, 1.6, 0.9, 0.2, 0.0, 1.0),
        ('SDXL', 1.3, 1.4, 0.9, 0.2, 0.0, 1.0),
    ]
    try:
        import FUM_presets
        presets = presets_builtin + FUM_presets.presets_custom
    except:
        presets = presets_builtin

    def title(self):
        return "FUM (FreeU-Move)"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        
        with InputAccordion(False, label=self.title(),
                          elem_id="extensions-FUM",
                          elem_classes=["extensions-FUM"]) as FUM_enabled:
            with gr.Row():
                FUM_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                FUM_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
            with gr.Row():
                FUM_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                FUM_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
            with gr.Row():
                FUM_start = gr.Slider(label='Start step', minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                FUM_end   = gr.Slider(label='End step', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
            with gr.Row():
                FUM_preset = gr.Dropdown(label='', choices=[x[0] for x in FUMForForge.presets], value='(presets)', type='index', scale=0, allow_custom_value=True)
            with gr.Row():
                random_move_checkbox = gr.Checkbox(label="Random UNet Move", value=False)
            with gr.Row():
                simple_move_s1_checkbox = gr.Checkbox(label="Simple UNet S1 Move", value=False)

        def setParams (preset):
            if preset < len(FUMForForge.presets):
                return  FUMForForge.presets[preset][1], FUMForForge.presets[preset][2], \
                        FUMForForge.presets[preset][3], FUMForForge.presets[preset][4], \
                        FUMForForge.presets[preset][5], FUMForForge.presets[preset][6], '(presets)'
            else:
                return 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, '(presets)'

        FUM_preset.input( fn=setParams,
                            inputs=[FUM_preset],
                            outputs=[FUM_b1, FUM_b2, FUM_s1, FUM_s2, FUM_start, FUM_end, FUM_preset], show_progress=False)

        self.infotext_fields = [
            (FUM_enabled, lambda d: d.get("FUM_enabled", False)),
            (FUM_b1,      "FUM_b1"),
            (FUM_b2,      "FUM_b2"),
            (FUM_s1,      "FUM_s1"),
            (FUM_s2,      "FUM_s2"),
            (FUM_start,   "FUM_start"),
            (FUM_end,     "FUM_end"),
            (random_move_checkbox, lambda d: d.get("random_move_enabled", False)),
            (simple_move_s1_checkbox, lambda d: d.get("simple_move_s1_enabled", False)),
        ]

        return FUM_enabled, FUM_b1, FUM_b2, FUM_s1, FUM_s2, FUM_start, FUM_end, random_move_checkbox, simple_move_s1_checkbox

    def denoiser_callback(self, params):
        thisStep = params.sampling_step / (params.total_sampling_steps - 1)
        
        if thisStep >= FUMForForge.FUM_start and thisStep <= FUMForForge.FUM_end:
            FUMForForge.doFUM = True
        else:
            FUMForForge.doFUM = False

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # If you use highres fix, this will be called twice.

        FUM_enabled, FUM_b1, FUM_b2, FUM_s1, FUM_s2, FUM_start, FUM_end, random_move_checkbox, simple_move_s1_checkbox = script_args

        if not FUM_enabled:
            return

        FUMForForge.random_move_enabled = random_move_checkbox
        FUMForForge.simple_move_s1_enabled = simple_move_s1_checkbox

        # Randomly adjust S1 and S2 if the option is enabled
        if FUMForForge.random_move_enabled:
            FUMForForge.last_s1 += random.choice([-0.01, 0.01])
            FUMForForge.last_s2 += random.choice([-0.01, 0.01])

            # Clamp values to avoid going out of the allowed range (0 to 4)
            FUMForForge.last_s1 = max(0, min(4, FUMForForge.last_s1))
            FUMForForge.last_s2 = max(0, min(4, FUMForForge.last_s2))

            FUM_s1 = FUMForForge.last_s1
            FUM_s2 = FUMForForge.last_s2

        unet = p.sd_model.forge_objects.unet

        # Simple increase S1 if the option is enabled
        if FUMForForge.simple_move_s1_enabled:
            FUMForForge.last_s1 += 0.002
            FUMForForge.last_s2 = FUMForForge.last_s2

            # Clamp values to avoid going out of the allowed range (0 to 4)
            FUMForForge.last_s1 = max(0, min(4, FUMForForge.last_s1))
            FUMForForge.last_s2 = max(0, min(4, FUMForForge.last_s2))

            FUM_s1 = FUMForForge.last_s1
            FUM_s2 = FUMForForge.last_s2

        unet = p.sd_model.forge_objects.unet

        #   test if patchable
        model_channels = unet.model.diffusion_model.config.get("model_channels")
        if model_channels is None:
            gr.Info("FUM is not supported for this model!")
            return

        FUMForForge.FUM_start = FUM_start
        FUMForForge.FUM_end   = FUM_end
        on_cfg_denoiser(self.denoiser_callback)

        unet = patch_FUM_v2(unet, FUM_b1, FUM_b2, FUM_s1, FUM_s2)

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        p.extra_generation_params.update(dict(
            FUM_enabled   = FUM_enabled,
            FUM_b1        = FUM_b1,
            FUM_b2        = FUM_b2,
            FUM_s1        = FUM_s1,
            FUM_s2        = FUM_s2,
            FUM_start     = FUM_start,
            FUM_end       = FUM_end,
            random_move_enabled = FUMForForge.random_move_enabled,
            simple_move_s1_enabled = FUMForForge.simple_move_s1_enabled,
        ))

        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return
