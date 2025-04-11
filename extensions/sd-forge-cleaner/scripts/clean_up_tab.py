import modules.scripts as scripts
import gradio as gr
from modules.shared import opts, OptionInfo
from modules import script_callbacks
from modules.ui_components import ResizeHandleRow
from scripts import lama
from PIL import Image
import numpy as np
from modules_forge.forge_canvas.canvas import ForgeCanvas, LogicalImage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def on_ui_settings():
    section = ('cleaner', "Cleaner")
    opts.add_option("cleaner_use_gpu", OptionInfo(True, "Use GPU", gr.Checkbox, {"interactive": True}, section=section))

def send_to_cleaner(result):
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "name" in result[0]:
        image = Image.open(result[0]["name"])
    else:
        image = result
    logger.info(f"Sending image back to cleaner: {image}")
    return image

def extract_mask(foreground):
    if foreground is None:
        return None
    if isinstance(foreground, np.ndarray):
        return Image.fromarray(foreground[:,:,3])
    return foreground.split()[3].convert('L')

def clean_wrapper(foreground, background):
    if background is None:
        return None
    
    mask = extract_mask(foreground)
    if mask is None:
        return None
    
    if isinstance(background, np.ndarray):
        background = Image.fromarray(background)
    
    background = background.convert("RGB")
    
    return lama.clean_object(background, mask)

def process_gallery_output(img):
    if img is None:
        return None
    # If img is already a tuple (image, caption), extract just the image
    if isinstance(img, tuple):
        img = img[0]
    return img

def on_ui_tabs():
    with gr.Blocks() as object_cleaner_tab:
        for tab_name in ["Clean up", "Clean up upload"]:
            with gr.Tab(tab_name) as clean_up_tab:
                with ResizeHandleRow(equal_height=False):
                    # Left column for input
                    with gr.Column(scale=1):
                        if tab_name == "Clean up":
                            init_img_with_mask = ForgeCanvas(
                                height=650,
                                elem_id="cleanup_img2maskimg",
                            )
                            inputs = [init_img_with_mask.foreground, init_img_with_mask.background]
                            clean_fn = clean_wrapper
                        else:
                            clean_up_init_img = gr.Image(
                                label="Image for cleanup", 
                                source="upload", 
                                type="pil", 
                                elem_id="cleanup_img_inpaint_base"
                            )
                            clean_up_init_mask = gr.Image(
                                label="Mask", 
                                source="upload", 
                                type="pil", 
                                elem_id="cleanup_img_inpaint_mask"
                            )
                            inputs = [clean_up_init_img, clean_up_init_mask]
                            clean_fn = lama.clean_object

                    # Right column for output
                    with gr.Column(scale=1):
                        clean_button = gr.Button("Clean Up", height=100)
                        result_gallery = gr.Gallery(
                            label='Output',
                            show_label=False,
                            elem_id="cleanup_gallery",
                            preview=True,
                            height=512
                        )

                        clean_button.click(
                            fn=lambda *args: None,  # Just clear the image
                            outputs=[result_gallery]
                        ).then(
                            fn=clean_fn,
                            inputs=inputs,
                            outputs=[result_gallery]
                        ).then(
                            fn=process_gallery_output,  # Process the output before displaying
                            inputs=[result_gallery],
                            outputs=[result_gallery]
                        )

    return (object_cleaner_tab, "Cleaner", "cleaner_tab"),

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)