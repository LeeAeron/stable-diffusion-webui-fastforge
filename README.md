# Stable Diffusion WebUI Forge
Custom build, optimized for performance.

Specific changes compared to official Forge WebUI:
- optimized for max performance on all systems, especially with low VRAM
- added main .bat menu with additional options to change memory management profile, installing some models etc.
- pre-installed and pre-configured useful extensions, including FluxTools v2, which work 'out-of-box'
- re-configured main configs and settings
- no memory overflow

I tried to save native simple structure of Forge and add more features, which will work out-of-box, without additional setting them: 'just unpack, launch and work'.

# Installing Forge
***NOTE:***

***- I'm supporting only Windows 10/11 PCs.***

**Just use one-click installation package (with git and python included, rest will be downloaded at first start, including pytorch and all needed modules)**

[*** Download One-Click Package (CUDA 12.1 + Pytorch 2.3.1, will be downloaded at first time)***](https://github.com/LeeAeron/stable-diffusion-webui-fastforge/releases/download/v1.01/stable-diffusion-webui-forge-custom-build_1.01.2025.04.11.7z)

After you're downloaded archive, uncompress it, and use `update.bat` to update, and use `START.bat` to run. [3]

In main menu you can see 9 steps:

- 1 - Launches Forge WebUI.
- 2 - Menu for download upscale models.
Script will download two separate models with Aria downloader and place it into their models folder.
Also, there's possibility to download rest most popular upscale models-script will download zip and unpack it into aprpropriate modoels folders.
- 3 - Menu for download and install pre-integrated aDetailer additional models.
- 4 - Menu for download and install NF4v2 model (by lllyasviel)
- 5 - Menu for download and install Flux VAE and Text Encoders (CLIPs)
- 6 - Menu for download and install Flux models for pre-integrated FluxTools: Canny, Fill, Deppth. [1]
- 7 - Menu for download model for pre-integrated Prompt Translator (2Gb, Facebook offline model).
- 8-  Menu for changing Memory optimization profile - specific profile for SD/SDXL/Flux/ALL profiles, pre-configured for low-VRAM PCs and for normal PCs.
- 9 - Menu for changing Memory Clear profile: 'Pure Forge' or 'Clear Always'.  [2]

**NOTE: 
 - [1] For Redux you can use usual Flux models: NF4, fp8/fp16/Dev GGUF/Schnell/Schnell GGUF versions. For Canny, Depth and Fill you can use their fp8/fp16/Dev GGUF versions.
 - [2] 'Clear Always' clears memory with clearing GPU and RAM cache, that hepls to prevent memory overflow while changing between lot of models and while batch upscaling.
 - [3] For Docker users: MAIN MENU not available due to it's in webui-user.bat!
 
[Changelog here](https://github.com/LeeAeron/stable-diffusion-webui-fastforge/blob/main/CHANGELOG.md)

[Additional repositories, used in build](https://github.com/LeeAeron/stable-diffusion-webui-fastforge/blob/main/additional_repositories_inside.md)

# Additional models available for download and install via .bat menu provided by:
- Flux.1 Dev NF4 v2: https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4
- Flux.1 Dev fp8: https://huggingface.co/datasets/LeeAeron/flux_controlnet
- Flux.1 Dev VAE, CLIPS: https://huggingface.co/datasets/LeeAeron/flux_vae_encoders
- Facebook offline translate model: https://huggingface.co/datasets/LeeAeron/offline_translate_model


# FluxTools v2 for Dummies:

[![Video Title](https://img.youtube.com/vi/MHYSFBkF36s/0.jpg)](https://www.youtube.com/watch?v=MHYSFBkF36s)

Thanks for FluxTools v2 extension to https://github.com/AcademiaSD

Readme file, also changelog will be updated, and extended.
