#Changelog:

#2025/04/16
- added some repositories back to sources for install while first start
- deleted non-needed code from webui-user.bat about four mandatory repos check/install
- added Docker file for Docker run support
- some changes in requirements_versions

#2025/04/14
- enabled always show GPU Weights slider for SD/XL profiles (lowering weights helps while upscaling to great resolutions on LowRAM GPU)
- enabled ClipSkip slider for XL profile
- enlarged expand dimensions in Mosaic Outpaint extension
- fixes with webui-user.bat encoding (that crashes menu)

#2025/04/13
- enlarged FluxTools Fill outpaint expand max size to 2048px all sides (top/bottom/left/right)
- added some pre-confs into ui-config.json

#2025/04/11
* release for own Git repo
* moved main menu and some files from START.bat to webui folder and webui-user.bat for availability for Git users
* reworked python folder and requirements_versions, deleted some not not needed python modules
* fixed FluxTools Redux work
* ultralytics python module moved from pre-installed to requiremets
* reworked START.bat menu:
- deleted FlusTools Redux model part due to user can use now usual Flux.1 Dev fp8/fp16/GGUF and Flux Schnell
- replaced FluxTools Fill, Canny, Depth "open in browser" links to HigginFace repos with GGUFs for these models
- some code fixes for downloaded models
* some changes for webui-user.bat file
* Memory Management profile now set to "Always Clear Memory" by default
* deleted BiRefNet due to incompatible with new requiremets (dev updated to pytorch 2.5*)

#2025/04/09
* additional changes in main settings, UI conf, webui-user.bat files
* added additional keys with explanations into webui-user.bat file
* enlarged pictures size limits up to 4096x4096 for almost all
* re-configured 'sd-webui-prompt-format' extension: disabled 'remove underscores' by default
* deleted FluxTools module
* deleted FreeU module
* Integrated modules: 
- Aspect Ratio
- FluxTools v2, workable on LOW VRAM (tested with GTX1660Ti Mobile 6Gb + 64Gb RAM)
- additional samplers
- seamless inpainting
- replacement for FreeU
* small changes in START.bat menu:
- deleted text_encoders_FP8.zip download option, due to no need now
- added option (step (9)) to change MEMORY MANAGENT config: Native ForgeSD/Always Clear Memory (this helps to prevent VRAM/RAM overflow)
- fixes in menu code
* reworked RAM optimizations profiles:
- now there are normal ForgeSD profile, optimized with changed picture dimensions, and TOTAL_VRAM-1GB/-2GB 
- TOTAL_VRAM-3Gb profile has been deleted
- also, raised WEIGHTS for SD profile (SD/SDXL/FLUX/ALL) in all optimized profiles

#SOME EXPLANATIONS:
- if you're used previous version, just move old models into new, except 'models\diffusers' folder.
New FluxTools will download needed filess when you will use Fill, Redux, Canny first time (it's about 3.6Gb).
Main Cann, Redux, Fill, Depth models are same as before.

- if you have 6-8Gb VRAM and 32-128Gb RAM you can also use new FluxTools:
choose needed model (for example Flux Fill for outpaint with Fill), mode WEIGHTS SLIDER to 1Gb, and change memory profile to ASYNC+CPU.
Generation will take some time, but it will work for you. Tested on my laptop with GTX1660Ti Mobile 6Gb + 64Gb RAM.

- If you had some VRAM and RAM overflow problems, go to step 9 in main START.bat menu and choose ALWAYS CLEAR MEMORY.
This may help with overflow.

#2025/04/03
* optimized rewuirements list
* deleted and updated some python modules
* reconfigured some settings by default in config file
* added additional menu in to bat file: download Flux VAE, Flux CLIP I, CLIP II (FP16-based, universal), and CLIP I Detailed for more detailis.
Links provided by my HugginFace cloud folder.
* fixed unpack error for archive

#2025/04/02
* torch moved to requirements instead of preconfigured in build (Forge will download Torch while first start and configure it to fit your PC)
* moved onnxruntime to real GPU version, also moved to requirements instead of pre-congigured in build
* moved bitsandbytes to requirements instead of pre-congigured in build
* updated PIP

#2025/03/30
* added additional upscalers pack download option into main menu/upscalers download
OLD VERSION USERS CAN ADD NEW MENU BY REPLACING AN OLD START.bat with NEW ONE!
* added empty Lora folder by default
* added Civit.Ai extension (https://civitai.com/user/TomDom)
* reworked RAM optimization profiles (main menu, step (7)):
- moved to Queue+CPU and Async+CPU profiles as most stable and without RAM/VRAM leak
OLD VERSION USERS CAN REPLACE PROFILES BY DELETING OLD AND COPYING WEBUI\ram_opt FOLDER!

#2025/03/29
* reworked python requirements, and inbuild modules, optimized code
* fixed RAM/VRAM leakage in memory optmized profiles (in MAIN MENU, step (7)). Now working fine with SD/XL/IL/FLUX.
* raised speed generation for SD/SDXL/IL/FLUX, especially for Flux fp8 and greatly for Flux fp16 models (from 15 minutes up to to 6-7 minutes on GTX 1660Ti Mobile 6Gb)
* disabled Auto-update changes option by default in Prompt Translate module
* added DenOfEquity's HyperTile extension, adapted to last Forge sources (https://github.com/DenOfEquity/HyperTile)

#2025/03/26
* moved from curl onto aria2c downloader, thx to @NeuroDonu for info
* reworked start script, extended menu
* added option download nf4v2 model by script (aria2c) or open in Browser
* added AcademiaSD' Flux ControlNet extension modded by @li_aeron (me). WARN! Work with 8Gb VRAM and up!
- You have to install all needed models, also download text_encoders_FP8.zip via MAIN MENU. 
Optionally, you can create HugginFace access token and place it into file webui\huggingface_access_token.txt, but it's not necessary in my modded FLux Tools extension. 
* reworked requirements_versions to fit new build with Flux ControlNet extension
* in MAIN MENU there was added new menu to download models and text_encoders_FP8 zip for Flux ControlNet, with choice to download by own or open in Browser
* some modules code optimization
* added Physton prompt translator extension (https://github.com/Physton/sd-webui-prompt-all-in-one prompt) with option to download pretrained offline (facebook) model for it in MAIN MENU

#2025/03/25
* reworked start script, now it has own menu:
*** ability to download two upscale models downloaded at first start from hugginface/LeeAeron (downloaded into models/DAT)
*** ability to download 8 additional adetailer models downloaded at first start from hugginface/LeeAeron (downloaded into models/adetailer)
*** ability to download Flux.1 Dev NF4v2 model with inbuilt VAE and CLIP models (11gb) (downloaded into models/Stable-diffusion)
*** ability to reconfiigure RAM optimizations profiles: pure ForgeSD/optimized (normal) ForgeSD/for PC with high RAM low VRAM (Async+Shared/Queue+Shared)
* reworked system folder, now very close to official Forge SD, this helps to make unpacked build weight less up to 0.5Gb
* now build will download and install inside itself all needed python modules, including xformers
* cleared some files as no needed
* deleted some warns and notif messages in Gradio and Detail Daemon
* added some performance optimizations into web-user.bat file
* pure Forge SD RAM/image dimensions by default

#2025/03/22
* added Digiearts sd-forge-cleaner extension
* added DenOfEquity superPrompter-webUI extension
* added Haoming02 sd-webui-prompt-format extension

#2025/03/09
* updated Git sources 2025/03/09
* reworked some Python modules and files
* fixed ControlNet work for SD/SDXL/IL models (OpenPose etc.)
* deleted some unneeded files

#2025/03/01
* updated Git sources 2025/02/28
* deleted RealESRGAN_x4plus, ScuNET, SwinIR_4x as useless
* deleted duplicated embeddings for sd/xl (negative etc)
* deleted some empty folders
* fixed auto-setup VRAM lvl for profiles by default

#2025/02/27
* forked from official Forge Git stable build with CUDA 12.1 and Pytorch 2.3.1
* included all last updates from Git by 2025/02/27
* added and enabled working xformers
* enabled Cuda Stream, Cuda Malloc for engine
* added Adetailer models (all that I found in inretnet) and replaced cached models to files itself, this make faster UI launch
* added ADetailer 'Eyes' model
* deleted nsfw / watermark checker code in diffusers Python module (in Portable Version)
* added img2img HiRes Fix. fully working version
* added TeaCache
* added 3D openPose Editor
* added Detail Demon. Own pre-configured version
* added Mosaic Outpaint
* added SD Upscale
* added 4xFFHQDAT, 4xSSDIRDAT, RealESRGAN_x4plus, ScuNET, SwinIR_4x upscale models
* pre-configured configs and settings
* added some embeddings
* unlocked deleted 'Flux Realistic' samplers with pre-integrated Google code for Flux.D / Flux.S models
* re-linked all chanes to intermal webui_cache folder for clearer and better engine work
