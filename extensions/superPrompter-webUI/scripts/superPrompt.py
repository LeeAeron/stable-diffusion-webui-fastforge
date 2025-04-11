import gradio

import os
import torch
import re
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from modules import scripts, shared
from modules.processing import get_fixed_seed


class SuprPromptr(scripts.Script):
    def __init__(self):
        self.tokenizer = None
        self.superprompt = None

    def title(self):
        return "SuperPrompt"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def remove_incomplete_sentence(self, paragraph):
        return re.sub(r'((?:\[^.!?\](?!\[.!?\]))\*+\[^.!?\\s\]\[^.!?\]\*$)', '', paragraph.rstrip())

    def generate(self, prompt, seed, tokenCount, punish, purge):
        tokenizer = getattr (shared, 'SuperPrompt_tokenizer', None)
        superprompt = getattr (shared, 'SuperPrompt_model', None)
        if tokenizer is None:
            tokenizer = T5TokenizerFast.from_pretrained(
                'roborovski/superprompt-v1',
            )
            shared.SuperPrompt_tokenizer = tokenizer
        if superprompt is None:
            superprompt = T5ForConditionalGeneration.from_pretrained(
                'roborovski/superprompt-v1',
                device_map='auto',
                torch_dtype=torch.float16
            )
            shared.SuperPrompt_model = superprompt
            print("SuperPrompt-v1 model loaded successfully.")
            if torch.cuda.is_available():
                superprompt.to('cuda')

        torch.manual_seed(get_fixed_seed(seed))
        systemprompt1 = "Expand the following prompt to add more detail: "
        
        input_ids = tokenizer(systemprompt1 + prompt, return_tensors="pt").input_ids.to(superprompt.device)
        
        outputs = superprompt.generate(input_ids, max_new_tokens=tokenCount, repetition_penalty=punish, do_sample=True)
        
        dirty_text = tokenizer.decode(outputs[0])
        result = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
        
        if purge:
            result = self.remove_incomplete_sentence(result)

        return result
        
    def unloadModel(self):
        del self.superprompt, self.tokenizer
        self.superprompt = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        return

    def ui(self, *args, **kwargs):
        with gradio.Accordion(open=False, label=self.title()):
            smolprompt = gradio.Textbox(label='Prompt', placeholder='Enter a prompt here...', value='', lines=1.01)
            biigprompt = gradio.Textbox(label='Result', value='', lines=1.01, show_copy_button=True, show_label=True)
            with gradio.Row():
                randomSeed = gradio.Number(label="seed", value=14641, precision=0, scale=1)
                tokenCount = gradio.Number(label="max new tokens", minimum=32, maximum=1024, value=256, precision=0, scale=1)
                repetition = gradio.Slider(label="repetition penalty", minimum=0.0, maximum=2.0, value=1.2, step=0.01, scale=1)
                incomplete = gradio.Checkbox(label="remove partial sentences", value=True, scale=1)
            with gradio.Row():
                goFirst = gradio.Button(value="Get yer fancy words", scale=2)
                goAgain = gradio.Button(value="Reprocess result", scale=1)
            with gradio.Row():
                unload = gradio.Button(value="unload model", scale=0)

    ##  buttons to get main prompt, send main prompt

        goFirst.click(fn=self.generate, inputs=[smolprompt, randomSeed, tokenCount, repetition, incomplete], outputs=[biigprompt])    
        goAgain.click(fn=self.generate, inputs=[biigprompt, randomSeed, tokenCount, repetition, incomplete], outputs=[biigprompt])    
        unload.click(fn=self.unloadModel, inputs=[], outputs=[])    

        return


