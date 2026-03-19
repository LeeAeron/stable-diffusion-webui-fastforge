@echo off
:main_menu
cls
echo ====================================================================
echo                 FastForge StableDiffusion Main Menu
echo ====================================================================
echo 1. Start FastForge StableDiffusion (Normal)
echo 2. Start FastForge StableDiffusion (RTX50xx)
echo 3. Check and download upscale models
echo 4. Check and download additional adetailer models
echo 5. Download Flux.1 Dev NF4 v2 model
echo 6. Download Flux.1 Kontext models
echo 7. Download Flux.1 Dev VAE and Encoders (CLIP)
echo 8. Download Flux.1 models for FluxTools
echo 9. Download Prompt Translate Extension (with offline model included)
echo 10. Download ClarityHD Upscale Base Model + Loras
echo 11. Update FastForge
echo ====================================================================
echo @https://t.me/li_aeron
echo Made in Ukraine
echo https://github.com/LeeAeron/stable-diffusion-webui-fastforge
echo ====================================================================
set /p choice=Choose action 1-11:
if "%choice%"=="1" goto start_forge
if "%choice%"=="2" goto start_forge_rtx50xx
if "%choice%"=="3" goto check_and_install_upscale_models
if "%choice%"=="4" goto check_and_install_adetailer_models
if "%choice%"=="5" goto download_fluxd_nf4_mn
if "%choice%"=="6" goto download_fluxd_kontext
if "%choice%"=="7" goto download_fluxd_vae_menu
if "%choice%"=="8" goto menu_controlnet
if "%choice%"=="9" goto menu_offline_transl
if "%choice%"=="10" goto clarityhd_models
if "%choice%"=="11" goto update_fastforge
echo Wrong choice. Please, try again.
pause
goto main_menu

:check_and_install_upscale_models
cls
echo ================================================================
echo       			    Upscale models download
echo ================================================================
echo 1. Install 4xFFHQDAT upscale model
echo 2. Install 4xSSDIRDAT upscale model
echo 3. Download and install additional upscale models pack (3.3Gb)
echo 4. Download additional upscale models pack (3.3Gb) with Browser
echo 5. Download IMG2IMG Jaggernaut SD model
echo 6. Back
echo ================================================================
set /p file_choice=Choose action 1-6:
if "%file_choice%"=="1" goto check_and_install_4xFFHQDAT_models
if "%file_choice%"=="2" goto check_and_install_4xSSDIRDAT_models
if "%file_choice%"=="3" goto download_additional_upscale_models
if "%file_choice%"=="4" goto open_with_browser_additional_upscale_models
if "%file_choice%"=="5" goto download_img2img_jaggernaut
if "%file_choice%"=="6" goto main_menu
echo Wrong choice. please, try again.
pause
goto check_and_install_upscale_models

:check_and_install_4xFFHQDAT_models
cls
echo Checking upscale models presence...
set "file_name=4xFFHQDAT.pth"
set "url=https://huggingface.co/datasets/LeeAeron/upscale_models/resolve/main/4xFFHQDAT.pth?download=true"
set "folder_path=.\models\DAT"
set "download_folder=models\DAT"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Upscale model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "4xFFHQDAT.pth" "%url%"
    echo Upscale model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto check_and_install_upscale_models

:check_and_install_4xSSDIRDAT_models
cls
set "file_name=4xSSDIRDAT.pth"
set "url=https://huggingface.co/datasets/LeeAeron/upscale_models/resolve/main/4xSSDIRDAT.pth?download=true"
set "folder_path=.\models\DAT"
set "download_folder=models\DAT"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Upscale model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "4xSSDIRDAT.pth" "%url%"
    echo Upscale model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto check_and_install_upscale_models

:download_additional_upscale_models
cls
echo Downloading additional upscalers models pack...
set "url=https://huggingface.co/datasets/LeeAeron/upscale_models/resolve/main/ADDITIONAL_UPSCALERS.zip?download=true"
set "download_folder=tmp"
set "extract_folder=models"
set "file_name=4xFaceUpDAT.pth"
set "folder_path=.\models\DAT\"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Additional upscalers models pack absent. Starting download...
	if not exist "%download_folder%" mkdir "%download_folder%"
	if not exist "%extract_folder%" mkdir "%extract_folder%"
	"%aria2_path%" -d "%download_folder%" -o "archive.zip" "%url%"
	powershell -Command "Expand-Archive -Path '%download_folder%\archive.zip' -DestinationPath '%extract_folder%' -Force"
	del "%download_folder%\archive.zip"
    echo Additional upscalers models pack succesfully downloaded.
) else (
echo OK.
)
pause
goto check_and_install_upscale_models

:open_with_browser_additional_upscale_models
cls
echo Opening additional upscalers models pack in Browser...
echo Unpack ZIP with 7zip and place folder or files itself into webui\models\ESRGAN or DAT folder.
set "url=https://huggingface.co/datasets/LeeAeron/upscale_models/resolve/main/ADDITIONAL_UPSCALERS.zip?download=true"
start "" "%url%"
)
pause
goto check_and_install_upscale_models

:download_img2img_jaggernaut
cls
set "file_name=IMG2IMG_JuggernautReborn.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/upscale_models/resolve/main/IMG2IMG_JuggernautReborn.safetensors?download=true"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Upscale model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "IMG2IMG_JuggernautReborn.safetensors" "%url%"
    echo Upscale model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto check_and_install_upscale_models

:check_and_install_adetailer_models
cls
echo Checking additional adetailer models presence...

set "file_name=deepfashion2_yolov8s-seg.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/deepfashion2_yolov8s-seg.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "deepfashion2_yolov8s-seg.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=Eyes.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/Eyes.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "Eyes.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=face_yolov8m.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/face_yolov8m.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "face_yolov8m.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=face_yolov8n_v2.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/face_yolov8n_v2.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "face_yolov8n_v2.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=face_yolov9c.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/face_yolov9c.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "face_yolov9c.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=hand_yolov8s.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/hand_yolov8s.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "hand_yolov8s.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=hand_yolov9c.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/hand_yolov9c.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "hand_yolov9c.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=person_yolov8m-seg.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/person_yolov8m-seg.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "person_yolov8m-seg.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto main_menu

:download_fluxd_nf4_mn
cls
echo ===============================
echo        Flux NF4v2 Model
echo ===============================
echo 1. Download NF4v2 model (11Gb)
echo 2. Open/download in Browser
echo 3. Back
echo ===============================
set /p file_choice=Choose action 1-3: 
if "%file_choice%"=="1" goto download_fluxd_nf4
if "%file_choice%"=="2" goto download_fluxd_nf4_browser
if "%file_choice%"=="3" goto main_menu
echo Wrong choice. please, try again.
pause
goto download_fluxd_nf4_mn

:download_fluxd_nf4
cls
set "file_name=flux1-dev-bnb-nf4-v2.safetensors"
set "url=https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4-v2.safetensors"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo flux1-dev-bnb-nf4-v2 model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "flux1-dev-bnb-nf4-v2.safetensors" "%url%"
    echo flux1-dev-bnb-nf4-v2 model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_nf4_mn

:download_fluxd_nf4_browser
cls
echo Opening Flux NF4v2 in Browser...
echo Place downloaded model into webui\models\Stable-diffusion folder.
echo This model already contain injected VAE and Text Encoders (Clips), so you don't need to use any external.
set "url=https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4-v2.safetensors"
start "" "%url%"
)
pause
goto download_fluxd_nf4_mn

:download_fluxd_kontext
cls
echo =============================================================
echo    		       Flux.1 Kontext Models
echo =============================================================
echo 1. Download Flux.1 Kontext fp16 (23Gb)
echo 2. Download Flux.1 Kontext fp8 e4m3fn (11Gb)
echo 3. Download Flux.1 Kontext fp8 e5m2 (11Gb)
echo 4. Download Flux.1 Kontext GGUF models via Browser
echo 5. Back
echo =============================================================
set /p file_choice=Choose action 1-5: 
if "%file_choice%"=="1" goto download_fluxd_kontext_fp16
if "%file_choice%"=="2" goto download_fluxd_kontext_fp8_e4m3fn
if "%file_choice%"=="3" goto download_fluxd_kontext_fp8_e5m2
if "%file_choice%"=="4" goto download_fluxd_kontext_gguf
if "%file_choice%"=="5" goto main_menu
echo Wrong choice. please, try again.
pause
goto download_fluxd_kontext

:download_fluxd_kontext_fp16
cls
set "file_name=flux1-kontext-dev.safetensors"
set "url=https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/resolve/main/flux1-kontext-dev.safetensors"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Flux.1 Kontext fp16 model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "flux1-kontext-dev.safetensors" "%url%"
    echo Flux.1 Kontext fp16 model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_kontext

:download_fluxd_kontext_fp8_e4m3fn
cls
set "file_name=FLUX.D_Kontext_fp8_e4m3fn.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/FluxKontext/resolve/main/FLUX.D_Kontext_fp8_e4m3fn.safetensors"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Flux.1 Kontext fp8 e4m3fn model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.D_Kontext_fp8_e4m3fn.safetensors" "%url%"
    echo Flux.1 Kontext fp8 e4m3fn model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_kontext

:download_fluxd_kontext_fp8_e5m2
cls
set "file_name=FLUX.D_Kontext_fp8_e5m2.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/FluxKontext/resolve/main/FLUX.D_Kontext_fp8_e5m2.safetensors"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Flux.1 Kontext fp8 e5m2 model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.D_Kontext_fp8_e5m2.safetensors" "%url%"
    echo Flux.1 Kontext fp8 e5m2 model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_kontext

:download_fluxd_kontext_gguf
cls
echo Opening Flux.1 Kontext GGUF models folder in Browser...
echo Place downloaded model(s) into webui\models\Stable-diffusion folder.
set "url=https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF/tree/main"
start "" "%url%"
)
pause
goto download_fluxd_kontext

:download_fluxd_vae_menu
cls
echo ================================================
echo     Flux.1 Dev VAE and Text Encoders (CLIP)
echo ================================================
echo 1. Download Flux.1 Dev VAE
echo 2. Download Flux.1 Dev Clip.l
echo 3. Download Flux.1 Dev Clip.l.Krea
echo 4. Download Flux.1 Dev Clip.l.ViT-L-14-BEST-smooth-GmP-HF-format
echo 5. Download Flux.1 Dev Clip.l.ViT-L-14-TEXT-detail-improved-hiT-GmP-HF
echo 6. Download Flux.1 Dev Clip.t5.v1.1.xxl-encoder_bf16
echo 7. Download Flux.1 Dev Clip.t5xxl_fp8.e4m3fn
echo 8. Download Flux.1 Dev Clip.t5xxl_fp16
echo 9. Download Chroma VAE model
echo 10. Download Illustrius Clip.l
echo 11. Download Illustrius Clip.g
echo 12. Open HugginFace folder in Browser
echo 13. Back
echo ================================================
set /p file_choice=Choose action 1-13: 
if "%file_choice%"=="1" goto download_vae
if "%file_choice%"=="2" goto download_clipl
if "%file_choice%"=="3" goto download_clipl_krea
if "%file_choice%"=="4" goto download_clipl_vit_L14_best
if "%file_choice%"=="5" goto download_clipl_vit_L14_text
if "%file_choice%"=="6" goto download_clip_t5_bf16
if "%file_choice%"=="7" goto download_clip_t5_fp8
if "%file_choice%"=="8" goto download_clip_t5_fp16
if "%file_choice%"=="9" goto download_chroma_vae
if "%file_choice%"=="10" goto download_ill_clipl
if "%file_choice%"=="11" goto download_ill_clipg
if "%file_choice%"=="12" goto download_vae_browser
if "%file_choice%"=="13" goto main_menu
echo Wrong choice. please, try again.
pause
goto download_fluxd_vae_menu

:download_vae
cls
set "file_name=FLUX.VAE.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.VAE.safetensors?download=true"
set "folder_path=.\models\vae"
set "download_folder=models\vae"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.VAE model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.VAE.safetensors" "%url%"
    echo FLUX.VAE model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clipl
cls
set "file_name=FLUX.Clip.l.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.Clip.l.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.Clip.l Text Encoder model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.Clip.l.safetensors" "%url%"
    echo FLUX.Clip.l Text Encoder model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clipl_krea
cls
set "file_name=FLUX.Clip.l.Krea.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.Clip.l.Krea.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.Clip.l.Krea Text Encoder model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.Clip.l.Krea.safetensors" "%url%"
    echo FLUX.Clip.l.Krea Text Encoder model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clipl_vit_L14_best
cls
set "file_name=FLUX.Clip.l.ViT-L-14-BEST-smooth-GmP-HF-format.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.Clip.l.ViT-L-14-BEST-smooth-GmP-HF-format.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.Clip.l.ViT-L-14-BEST-smooth-GmP-HF-format Text Encoder model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.Clip.l.ViT-L-14-BEST-smooth-GmP-HF-format.safetensors" "%url%"
    echo FLUX.Clip.l.ViT-L-14-BEST-smooth-GmP-HF-format Text Encoder model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clipl_vit_L14_text
cls
set "file_name=FLUX.Clip.l.ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.Clip.l.ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.Clip.l.ViT-L-14-TEXT-detail-improved-hiT-GmP-HF Text Encoder model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.Clip.l.ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors" "%url%"
    echo FLUX.Clip.l.ViT-L-14-TEXT-detail-improved-hiT-GmP-HF Text Encoder model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clip_t5_fp8
cls
set "file_name=FLUX.Clip.t5xxl_fp8.e4m3fn.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.Clip.t5xxl_fp8.e4m3fn.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.Clip.t5xxl_fp8.e4m3fn Text Encoder model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.Clip.t5xxl_fp8.e4m3fn.safetensors" "%url%"
    echo FLUX.Clip.t5xxl_fp8.e4m3fn Text Encoder model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clip_t5_fp16
cls
set "file_name=FLUX.Clip.t5xxl_fp16.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.Clip.t5xxl_fp16.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.Clip.t5xxl_fp16 Text Encoder model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.Clip.t5xxl_fp16.safetensors" "%url%"
    echo FLUX.Clip.t5xxl_fp16 Text Encoder model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clip_t5_bf16
cls
set "file_name=FLUX.Clip.t5.v1.1.xxl-encoder_bf16.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.Clip.t5.v1.1.xxl-encoder_bf16.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.Clip.t5.v1.1.xxl-encoder_bf16 Text Encoder model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "FLUX.Clip.t5.v1.1.xxl-encoder_bf16.safetensors" "%url%"
    echo FLUX.Clip.t5.v1.1.xxl-encoder_bf16 Text Encoder model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_chroma_vae
cls
set "file_name=diffusion_pytorch_model.safetensors"
set "url=https://huggingface.co/lodestones/Chroma/resolve/main/vae/diffusion_pytorch_model.safetensors"
set "folder_path=.\backend\huggingface\Chroma\vae"
set "download_folder=backend\huggingface\Chroma\vae"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Chroma VAE model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "diffusion_pytorch_model.safetensors" "%url%"
    echo Chroma VAE model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_ill_clipl
cls
set "file_name=IL.Clip.l.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/IL.Clip.l.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo IL.Clip.l Text Encoder model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "IL.Clip.l.safetensors" "%url%"
    echo IL.Clip.l Text Encoder model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_ill_clipg
cls
set "file_name=IL.Clip.g.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/IL.Clip.g.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo IL.Clip.g Text Encoder model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "IL.Clip.g.safetensors" "%url%"
    echo IL.Clip.g Text Encoder model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_vae_browser
cls
echo Opening HugginFace VAE/CLIP folder in Browser...
echo Place VAE file(s) into webui\models\vae folder.
echo Place Text Encoders (CLIP) file(s) into webui\models\text_encoder folder.
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders"
start "" "%url%"
)
pause
goto download_fluxd_vae_menu

:menu_controlnet
cls
echo ==============================================================
echo 				      FluxTools Models
echo ==============================================================
echo 1. Download FluxTools Canny model
echo 2. Download FluxTools Depth model
echo 3. Download FluxTools Fill model
echo 4. Back to main menu
echo ==============================================================
echo NOTE: 
echo 1. For FluxTools Redux you can use normal Flux model,
echo including Flux.1 Dev fp8/fp16, Flux.1 Dev GGUF, Flux Schell,
echo Flux Schnell GGUF and Flux.1 Dev NF4.
echo 2. For FluxTools Fill, Canny, Depth you can use fp8/fp16/GGUF.
echo ==============================================================
set /p file_choice=Choose action 1-4: 
if "%file_choice%"=="1" goto flux_controlnet_canny_menu
if "%file_choice%"=="2" goto flux_controlnet_depth_menu
if "%file_choice%"=="3" goto flux_controlnet_f_menu
if "%file_choice%"=="4" goto main_menu
echo Wrong choice. please, try again.
pause
goto menu_controlnet

:flux_controlnet_canny_menu
cls
echo =======================================================
echo               FluxTools Canny Model
echo =======================================================
echo 1. Download FluxTools Canny model (11Gb)
echo 2. Open/download FluxTools Canny GGUFs repo in Browser
echo 3. Back
echo =======================================================
set /p file_choice=Choose action 1-3: 
if "%file_choice%"=="1" goto flux_controlnet_canny
if "%file_choice%"=="2" goto flux_controlnet_canny_browser
if "%file_choice%"=="3" goto menu_controlnet
echo Wrong choice. please, try again.
pause
goto flux_controlnet_canny_menu

:flux_controlnet_canny
cls
echo Downloading FluxTools Canny...
set "file_name=flux1-Canny-Dev_FP8.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_controlnet/resolve/main/flux1-Canny-Dev_FP8.safetensors?download=true"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FluxTools Canny model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "flux1-Canny-Dev_FP8.safetensors" "%url%"
    echo FluxTools Canny model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto flux_controlnet_canny_menu

:flux_controlnet_canny_browser
cls
echo Opening FluxTools Canny HugginFace repo in Browser...
echo Place downloaded model into webui\models\Stable-diffusion folder.
set "url=https://huggingface.co/second-state/FLUX.1-Canny-dev-GGUF/tree/main"
start "" "%url%"
)
pause
goto flux_controlnet_canny_menu

:flux_controlnet_depth_menu
cls
echo =======================================================
echo    		    FluxTools Depth Model
echo =======================================================
echo 1. Download FluxTools Depth model (11Gb)
echo 2. Open/download FluxTools Depth GGUFs repo in Browser
echo 3. Back
echo =======================================================
set /p file_choice=Choose action 1-3: 
if "%file_choice%"=="1" goto flux_controlnet_depth
if "%file_choice%"=="2" goto flux_controlnet_depth_browser
if "%file_choice%"=="3" goto menu_controlnet
echo Wrong choice. please, try again.
pause
goto flux_controlnet_depth_menu

:flux_controlnet_depth
cls
echo Downloading FluxTools Depth...
set "file_name=flux1-Depth-Dev_FP8.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_controlnet/resolve/main/flux1-Depth-Dev_FP8.safetensors?download=true"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FluxTools Depth model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "flux1-Depth-Dev_FP8.safetensors" "%url%"
    echo FluxTools Depth model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto flux_controlnet_depth_menu

:flux_controlnet_depth_browser
cls
echo Opening FluxTools Depth GGUFs HugginFace repo in Browser...
echo Place downloaded model into webui\models\Stable-diffusion folder.
set "url=https://huggingface.co/SporkySporkness/FLUX.1-Depth-dev-GGUF/tree/main"
start "" "%url%"
)
pause
goto flux_controlnet_depth_menu

:flux_controlnet_f_menu
cls
echo ======================================================
echo     			 FluxTools Fill Model
echo ======================================================
echo 1. Download FluxTools Fill model (11Gb)
echo 2. Open/download FluxTools Fill GGUFs repo in Browser
echo 3. Back
echo ======================================================
set /p file_choice=Choose action 1-3: 
if "%file_choice%"=="1" goto flux_controlnet_fil
if "%file_choice%"=="2" goto flux_controlnet_fil_browser
if "%file_choice%"=="3" goto menu_controlnet
echo Wrong choice. please, try again.
pause
goto flux_controlnet_f_menu

:flux_controlnet_fil
cls
echo Downloading FluxTools Fill...
set "file_name=flux1-Fill-Dev_FP8.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_controlnet/resolve/main/flux1-Fill-Dev_FP8.safetensors?download=true"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FluxTools Fill model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "flux1-Fill-Dev_FP8.safetensors" "%url%"
    echo FluxTools Fill model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto flux_controlnet_f_menu

:flux_controlnet_fil_browser
cls
echo Opening FluxTools Fill HugginFace repo in Browser...
echo Place downloaded model into webui\models\Stable-diffusion folder.
set "url=https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-gguf/tree/main"
start "" "%url%"
)
pause
goto flux_controlnet_f_menu

:menu_offline_transl
cls
echo =========================================================================================
echo   		   Prompt Translate Extension (with offline model included)
echo =========================================================================================
echo 1. Download and install Prompt Translate Extension (with offline model included) (1.3Gb)
echo 2. Download Prompt Translate Extension (with offline model included) via Browser
echo 3. Back to main menu
echo =========================================================================================
set /p file_choice=Choose action 1-3: 
if "%file_choice%"=="1" goto facebook_translate_download
if "%file_choice%"=="2" goto facebook_translate_browser
if "%file_choice%"=="3" goto main_menu
echo Wrong choice. please, try again.
pause
goto menu_offline_transl

:facebook_translate_download
cls
echo Downloading Prompt Translate Extension (with offline model included)...
set "url=https://huggingface.co/datasets/LeeAeron/offline_translate_model/resolve/main/sd-webui-prompt-all-in-one.zip?download=true"
set "download_folder=tmp"
set "extract_folder=extensions"
set "file_name=main"
set "folder_path=.\extensions\sd-webui-prompt-all-in-one\models\models--facebook--mbart-large-50-many-to-many-mmt\refs\"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Facebook Prompt Offline Translate Extension %file_name% absent. Starting download...
	if not exist "%download_folder%" mkdir "%download_folder%"
	if not exist "%extract_folder%" mkdir "%extract_folder%"
	"%aria2_path%" -d "%download_folder%" -o "archive.zip" "%url%"
	powershell -Command "Expand-Archive -Path '%download_folder%\archive.zip' -DestinationPath '%extract_folder%' -Force"
	del "%download_folder%\archive.zip"
    echo Facebook Prompt Offline Translate Extensionl %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto menu_offline_transl

:facebook_translate_browser
cls
echo Opening Prompt Translate Extension (with offline model included) in Browser...
echo Unpack ZIP with 7zip and place 'sd-webui-prompt-all-in-one' folder into webui\extensions folder.
set "url=https://huggingface.co/datasets/LeeAeron/offline_translate_model/resolve/main/sd-webui-prompt-all-in-one.zip?download=true"
start "" "%url%"
)
pause
goto menu_offline_transl

:clarityhd_models
cls
echo Checking ClarityHD Upscaler base model and LoRas presence...

set "file_name=IMG2IMG_JuggernautReborn.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/upscale_models/resolve/main/IMG2IMG_JuggernautReborn.safetensors?download=true"
set "folder_path=.\models\checkpoints"
set "download_folder=models\checkpoints"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo ClarityHD Upscaler base model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "IMG2IMG_JuggernautReborn.safetensors" "%url%"
    echo ClarityHD Upscaler base model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=SD_FX_MoreDetails.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/LoRa/resolve/main/SD/SD_FX_MoreDetails.safetensors?download=true"
set "folder_path=.\models\Lora"
set "download_folder=models\Lora"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo SD_FX_MoreDetails LoRa model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "SD_FX_MoreDetails.safetensors" "%url%"
    echo SD_FX_MoreDetails LoRa model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=SDXLrender_v2.0.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/LoRa/resolve/main/SD/SDXLrender_v2.0.safetensors?download=true"
set "folder_path=.\models\Lora"
set "download_folder=models\Lora"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo SDXLrender_v2.0 LoRa model %file_name% absent. Starting download...
    "%aria2_path%" -d "%download_folder%" -o "SDXLrender_v2.0.safetensors" "%url%"
    echo SDXLrender_v2.0 LoRa model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto main_menu

:start_forge
cls
call run_normal.bat

:start_forge_rtx50xx
cls
call run_rtx50.bat

:update_fastforge
call ../environment.bat

for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD') do set CURRENT_BRANCH=%%i
git fetch origin
git reset --hard origin/%CURRENT_BRANCH%

echo [INFO] Update finished.
pause
goto main_menu

