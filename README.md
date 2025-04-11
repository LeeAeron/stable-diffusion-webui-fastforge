# Stable Diffusion WebUI Forge
Custom build, optimized for performance.

Specific changes compared to official Forge WebUI:
- optimized for max performance on all systems, especially with low VRAM
- added main .bat menu with additional options to change memory management profile, installing some models etc.
- pre-installed and pre-configured useful extensions, including FLuxUtils v2, which work 'out-of-box'
- re-configured main configs and settings
- no memory overflow

I tried to save native simple structure of Forge and add more features, which will work out-of-box, without additional setting them: 'just unpack, launch and work'.

# Installing Forge

**Just use one-click installation package (with git and python included, rest will be downloaded at first start, including pytorch and all needed modules)**

[*** Download One-Click Package (CUDA 12.1 + Pytorch 2.3.1, will be downloaded at first time)***](https://github.com/LeeAeron/stable-diffusion-webui-fastforge/releases/download/v1.01/stable-diffusion-webui-forge-custom-build_1.01.2025.04.11.7z)

After you're downloaded archive, uncompress it, and use `update.bat` to update, and use `START.bat` to run.

In main menu you can see 9 steps:

- 1 - Launches Forge WebUI.
- 2 - Menu for download upscale models.
Script will download two separate models with Aria downloader and place it into their models folder.
Also, there's possibility to download rest most popular upscale models-script will download zip and unpack it into aprpropriate modoels folders.
- 3 - Menu for download and install pre-integrated aDetailer additional models.
- 4 - Menu for download and install NF4v2 model (by lllyasviel)
- 5 - Menu for download and install FLux VAE and Text Encoders (CLIPs)
- 6 - Menu for download and install Flux models for pre-integrated FLuxTools: Canny, Fill, Deppth. [1]
- 7 - Menu for download model for pre-integrated Prompt Translator (2Gb, Facebook offline model).
- 8-  Menu for changing Memory optimization profile - specific profile for SD/SDXL/Flux/ALL profiles, pre-configured for low-VRAM PCs and for normal PCs.
- 9 - Menu for changing Memory Clear profile: 'Pure Forge' or 'Clear Always'.  [2]

**NOTE: 
 - [1] For Redux you can use usual Flux models: NF4, fp8/fp16/Dev GGUF/Schnell/Schnell GGUF versions. For Canny, Depth and Fill you can use their fp8/fp16/Dev GGUF versions.
 - [2] 'Clear Always' clears memory with clearing GPU and RAM cach/scum, that hepls to prevent memory overflow while changing between lot of models and while batch upscaling.

# Additional repositories used in build:

* Mandatory to install: 
- https://github.com/salesforce/BLIP
- https://github.com/lllyasviel/huggingface_guess
- https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets
- https://github.com/lllyasviel/google_blockly_prototypes

* Optional, pre-integrated, non-updatable:
- https://github.com/Bing-su/adetailer
- https://github.com/altoiddealer/--sd-webui-ar-plusplus
- https://github.com/silvertuanzi/advanced_euler_sampler_extension
- https://github.com/DenOfEquity/forgeFlux_dualPrompt
- https://github.com/DenOfEquity/HyperTile
- https://github.com/amadeus-ai/img2img-hires-fix
- https://github.com/Avaray/lora-keywords-finder
- https://civitai.com/models/151467/civitai-browser-or-sd-webui-extension
- https://github.com/novitalabs/sd-webui-cleaner
- https://github.com/Haoming02/sd-forge-couple
- https://github.com/AcademiaSD/sd-forge-fluxtools-v2
- https://github.com/zeittresor/sd-forge-fum
- https://github.com/likelovewant/sd-forge-teacache
- https://github.com/ZhUyU1997/open-pose-editor
- https://github.com/muerrilla/sd-webui-detail-daemon
- https://github.com/Haoming02/sd-webui-mosaic-outpaint
- https://github.com/Physton/sd-webui-prompt-all-in-one
- https://github.com/Haoming02/sd-webui-prompt-format
- https://github.com/licyk/sd-webui-tcd-sampler
- https://github.com/brick2face/seamless-tile-inpainting
- https://github.com/DenOfEquity/superPrompter-webUI

# Additional models provided from:
- NF4 v2: ttps://huggingface.co/lllyasviel
- FluxTools fp8: https://huggingface.co/LeeAeron
- FluxTools Canny GGUF: https://huggingface.co/second-state/FLUX.1-Canny-dev-GGUF
- FluxTools Depth GGUF: https://huggingface.co/SporkySporkness/FLUX.1-Depth-dev-GGUF
- FluxTools Fill GGUF: https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-gguf

# FluxTools v2 for Dummies: 
[![Watch the video](https://img.youtube.com/vi/MHYSFBkF36s/hqdefault.jpg)](https://www.youtube.com/watch?v=MHYSFBkF36s)

Very thanks to @AcademiaSD for extension.

Readme file, also changelog will be updated, and extended.
