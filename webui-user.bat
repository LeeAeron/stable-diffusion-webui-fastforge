@echo off
:main_menu
cls
echo =============================================================
echo                        Forge SD MENU
echo =============================================================
echo 1. Start Forge SD
echo 2. Check and download upscale models
echo 3. Check and download additional adetailer models
echo 4. Download Flux.1 Dev NF4 v2 model
echo 5. Download Flux.1 Dev VAE and Encoders (CLIP)
echo 6. Download Flux.D models for Flux ControlNet
echo 7. Download model for Prompt Translate
echo 8. Change RAM optimizations profile
echo 9. Change Memory Clear profile
echo =============================================================
echo Forked, modified by @li_aeron
echo https://github.com/LeeAeron/stable-diffusion-webui-fastforge
echo =============================================================
set /p choice=Choose action 1-9:
if "%choice%"=="1" goto start_forge
if "%choice%"=="2" goto check_and_install_upscale_models
if "%choice%"=="3" goto check_and_install_adetailer_models
if "%choice%"=="4" goto download_fluxd_nf4_mn
if "%choice%"=="5" goto download_fluxd_vae_menu
if "%choice%"=="6" goto menu_controlnet
if "%choice%"=="7" goto menu_offline_transl
if "%choice%"=="8" goto ram_opt
if "%choice%"=="9" goto memory_managmnt_prfl
echo Wrong choice. Please, try again.
pause
goto main_menu

:check_and_install_upscale_models
cls
echo ==========================================
echo         Upscale models download
echo ==========================================
echo 1. Install 4xFFHQDAT upscale model
echo 2. Install 4xSSDIRDAT upscale model
echo 3. Download and install additional upscale models pack (3.3Gb)
echo 4. Download additional upscale models pack (3.3Gb) with Browser
echo 5. Back
echo ===========================
set /p file_choice=Choose action 1-5:
if "%file_choice%"=="1" goto check_and_install_4xFFHQDAT_models
if "%file_choice%"=="2" goto check_and_install_4xSSDIRDAT_models
if "%file_choice%"=="3" goto download_additional_upscale_models
if "%file_choice%"=="4" goto open_with_browser_additional_upscale_models
if "%file_choice%"=="5" goto main_menu
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
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Upscale model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "4xFFHQDAT.pth" "%url%"
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
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Upscale model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "4xSSDIRDAT.pth" "%url%"
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
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Additional upscalers models pack absent. Starting download...
	if not exist "%download_folder%" mkdir "%download_folder%"
	if not exist "%extract_folder%" mkdir "%extract_folder%"
	aria2c -d "%download_folder%" -o "archive.zip" "%url%"
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
set "url=https://huggingface.co/datasets/LeeAeron/upscale_models/resolve/main/ADDITIONAL_UPSCALERS.zip?download=true"
start "" "%url%"
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
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "deepfashion2_yolov8s-seg.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=Eyes.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/Eyes.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "Eyes.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=face_yolov8m.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/face_yolov8m.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "face_yolov8m.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=face_yolov8n_v2.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/face_yolov8n_v2.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "face_yolov8n_v2.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=face_yolov9c.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/face_yolov9c.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "face_yolov9c.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=hand_yolov8s.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/hand_yolov8s.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "hand_yolov8s.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=hand_yolov9c.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/hand_yolov9c.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "hand_yolov9c.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)

set "file_name=person_yolov8m-seg.pt"
set "url=https://huggingface.co/datasets/LeeAeron/adetailer_models/resolve/main/person_yolov8m-seg.pt?download=true"
set "folder_path=.\models\adetailer"
set "download_folder=models\adetailer"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo adetailer model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "person_yolov8m-seg.pt" "%url%"
    echo adetailer model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto main_menu

:download_fluxd_nf4_mn
cls
echo =============================
echo      Flux NF4v2 Model
echo =============================
echo 1. Download NF4v2 model (11Gb)
echo 2. Open/download in Browser
echo 3. Back
echo =============================
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
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo flux1-dev-bnb-nf4-v2 model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "flux1-dev-bnb-nf4-v2.safetensors" "%url%"
    echo flux1-dev-bnb-nf4-v2 model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_nf4_mn

:download_fluxd_nf4_browser
cls
echo Opening Flux NF4v2 in Browser...
set "url=https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4-v2.safetensors"
start "" "%url%"
)
pause
goto download_fluxd_nf4_mn

:download_fluxd_vae_menu
cls
echo ==================================
echo Flux.1 Dev VAE and Encoders (CLIP)
echo ==================================
echo 1. Download Flux.1 Dev VAE (319Mb)
echo 2. Download Flux.1 Dev CLIP I (234Mb)
echo 3. Download Flux.1 Dev CLIP I Detailed (888Mb)
echo 4. Download Flux.1 Dev CLIP II Detailed (9.11Gb)
echo 5. Open HugginFace folder in Browser
echo 6. Back
echo ==================================
set /p file_choice=Choose action 1-6: 
if "%file_choice%"=="1" goto download_vae
if "%file_choice%"=="2" goto download_clip1
if "%file_choice%"=="3" goto download_clip1_detaled
if "%file_choice%"=="4" goto download_clip2
if "%file_choice%"=="5" goto download_vae_browser
if "%file_choice%"=="6" goto main_menu
echo Wrong choice. please, try again.
pause
goto download_fluxd_vae_menu

:download_vae
cls
set "file_name=FLUX.VAE.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.VAE.safetensors?download=true"
set "folder_path=.\models\vae"
set "download_folder=models\vae"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.VAE model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "FLUX.VAE.safetensors" "%url%"
    echo FLUX.VAE model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clip1
cls
set "file_name=FLUX.I.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.I.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.I model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "FLUX.I.safetensors" "%url%"
    echo FLUX.I model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clip1_detaled
cls
set "file_name=FLUX.I.Detailed.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.I.Detailed.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.I.Detailed model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "FLUX.I.Detailed.safetensors" "%url%"
    echo FLUX.I.Detailed model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_clip2
cls
set "file_name=FLUX.II.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders/resolve/main/FLUX.II.safetensors?download=true"
set "folder_path=.\models\text_encoder"
set "download_folder=models\text_encoder"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo FLUX.II model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "FLUX.II.safetensors" "%url%"
    echo FLUX.II model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto download_fluxd_vae_menu

:download_vae_browser
cls
echo Opening HugginFace VAE/CLIP folder in Browser...
set "url=https://huggingface.co/datasets/LeeAeron/flux_vae_encoders"
start "" "%url%"
)
pause
goto download_fluxd_vae_menu

:menu_controlnet
cls
echo ===========================
echo   Flux ControlNet Models
echo ===========================
echo 1. Download Canny model
echo 2. Download Depth model
echo 3. Download Fill model
echo 4. Back to main menu
echo ==============================================================
echo NOTE: 
echo 1. For FluxTools REDUX you can use normal Flux model,
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
echo ===========================
echo      Flux Canny Model
echo ===========================
echo 1. Download Canny model (11Gb)
echo 2. Open/download in Browser
echo 3. Back
echo ===========================
set /p file_choice=Choose action 1-3: 
if "%file_choice%"=="1" goto flux_controlnet_canny
if "%file_choice%"=="2" goto flux_controlnet_canny_browser
if "%file_choice%"=="3" goto menu_controlnet
echo Wrong choice. please, try again.
pause
goto flux_controlnet_canny_menu

:flux_controlnet_canny
cls
echo Downloading Flux Canny...
set "file_name=flux1-Canny-Dev_FP8.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_controlnet/resolve/main/flux1-Canny-Dev_FP8.safetensors?download=true"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Flux Canny model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "flux1-Canny-Dev_FP8.safetensors" "%url%"
    echo Flux Canny model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto flux_controlnet_canny_menu

:flux_controlnet_canny_browser
cls
echo Opening Flux Canny HugginFace repo in Browser...
set "url=https://huggingface.co/second-state/FLUX.1-Canny-dev-GGUF/tree/main"
start "" "%url%"
)
pause
goto flux_controlnet_canny_menu

:flux_controlnet_depth_menu
cls
echo ===========================
echo      Flux Depth Model
echo ===========================
echo 1. Download Depth model (11Gb)
echo 2. Open/download in Browser
echo 3. Back
echo ===========================
set /p file_choice=Choose action 1-3: 
if "%file_choice%"=="1" goto flux_controlnet_depth
if "%file_choice%"=="2" goto flux_controlnet_depth_browser
if "%file_choice%"=="3" goto menu_controlnet
echo Wrong choice. please, try again.
pause
goto flux_controlnet_depth_menu

:flux_controlnet_depth
cls
echo Downloading Flux Depth...
set "file_name=flux1-Depth-Dev_FP8.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_controlnet/resolve/main/flux1-Depth-Dev_FP8.safetensors?download=true"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Flux Depth model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "flux1-Depth-Dev_FP8.safetensors" "%url%"
    echo Flux Depth model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto flux_controlnet_depth_menu

:flux_controlnet_depth_browser
cls
echo Opening Flux Depth HugginFace repo in Browser...
set "url=https://huggingface.co/SporkySporkness/FLUX.1-Depth-dev-GGUF/tree/main"
start "" "%url%"
)
pause
goto flux_controlnet_depth_menu

:flux_controlnet_f_menu
cls
echo ===========================
echo      Flux Fill Model
echo ===========================
echo 1. Download Fill model (11Gb)
echo 2. Open/download in Browser
echo 3. Back
echo ===========================
set /p file_choice=Choose action 1-3: 
if "%file_choice%"=="1" goto flux_controlnet_fil
if "%file_choice%"=="2" goto flux_controlnet_fil_browser
if "%file_choice%"=="3" goto menu_controlnet
echo Wrong choice. please, try again.
pause
goto flux_controlnet_f_menu

:flux_controlnet_fil
cls
echo Downloading Flux Fill...
set "file_name=flux1-Fill-Dev_FP8.safetensors"
set "url=https://huggingface.co/datasets/LeeAeron/flux_controlnet/resolve/main/flux1-Fill-Dev_FP8.safetensors?download=true"
set "folder_path=.\models\Stable-diffusion"
set "download_folder=models\Stable-diffusion"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Flux Fill model %file_name% absent. Starting download...
    aria2c -d "%download_folder%" -o "flux1-Fill-Dev_FP8.safetensors" "%url%"
    echo Flux Fill model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto flux_controlnet_f_menu

:flux_controlnet_fil_browser
cls
echo Opening Flux Fill HugginFace repo in Browser...
set "url=https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-gguf/tree/main"
start "" "%url%"
)
pause
goto flux_controlnet_f_menu

:menu_offline_transl
cls
echo ==========================================
echo   Facebook Prompt Offline Translate Model
echo ==========================================
echo 1. Download and install Facebook offline model (1.3Gb)
echo 2. Download Facebook offline model via Browser
echo 3. Back to main menu
echo ===========================
set /p file_choice=Choose action 1-3: 
if "%file_choice%"=="1" goto facebook_translate_download
if "%file_choice%"=="2" goto facebook_translate_browser
if "%file_choice%"=="3" goto main_menu
echo Wrong choice. please, try again.
pause
goto menu_offline_transl

:facebook_translate_download
cls
echo Downloading Facebook offline translate model...
set "url=https://huggingface.co/datasets/LeeAeron/offline_translate_model/resolve/main/sd-webui-prompt-all-in-one.zip?download=true"
set "download_folder=tmp"
set "extract_folder=extensions\sd-webui-prompt-all-in-one\models"
set "file_name=main"
set "folder_path=.\extensions\sd-webui-prompt-all-in-one\models\models--facebook--mbart-large-50-many-to-many-mmt\refs\"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    echo Facebook offline translate model %file_name% absent. Starting download...
	if not exist "%download_folder%" mkdir "%download_folder%"
	if not exist "%extract_folder%" mkdir "%extract_folder%"
	aria2c -d "%download_folder%" -o "archive.zip" "%url%"
	powershell -Command "Expand-Archive -Path '%download_folder%\archive.zip' -DestinationPath '%extract_folder%' -Force"
	del "%download_folder%\archive.zip"
    echo Facebook offline translate model %file_name% succesfully downloaded.
) else (
echo OK.
)
pause
goto menu_offline_transl

:facebook_translate_browser
cls
echo Opening Facebook offline translate model in Browser...
set "url=https://huggingface.co/datasets/LeeAeron/offline_translate_model/resolve/main/sd-webui-prompt-all-in-one.zip?download=true"
start "" "%url%"
)
pause
goto menu_offline_transl

:ram_opt
cls
echo ===========================
echo  RAM optimizations profile
echo ===========================
echo 1. Set Pure Forge SD Profile and image dimensions
echo 2. Set profile for PC with 64-128Gb RAM and LOW VRAM
echo 3. Set profile with normal RAM and VRAM
echo 4. Back to main menu
echo ===========================
set /p file_choice=Choose action 1-4: 
if "%file_choice%"=="1" goto pure_forge
if "%file_choice%"=="2" goto highram_lowvram
if "%file_choice%"=="3" goto normal_ram_vram
if "%file_choice%"=="4" goto main_menu
echo Wrong choice. please, try again.
pause
goto ram_opt

:pure_forge
cls
set "file_name=main_entry.py"
set "source_folder=.\ram_opt\profiles_pure_forge"
set "destination_folder=.\modules_forge"
if exist "%source_folder%\%file_name%" (
    copy /y "%source_folder%\%file_name%" "%destination_folder%\%file_name%"
    echo Pure Forge SD profile succesfully applied.
) else (
    echo Profile absent.
)
pause
goto ram_opt

:normal_ram_vram
cls
set "file_name=main_entry.py"
set "source_folder=.\ram_opt\profiles_default"
set "destination_folder=.\modules_forge"
if exist "%source_folder%\%file_name%" (
    copy /y "%source_folder%\%file_name%" "%destination_folder%\%file_name%"
    echo Normal RAM and VRAM profile succesfully applied.
) else (
    echo Profile absent.
)
pause
goto ram_opt

:highram_lowvram
cls
echo ==========================================
echo  64-128Gb RAM and LOW VRAM profile choice
echo ==========================================
echo 1. Total VRAM - 1GB by default
echo 2. Total VRAM - 2GB by default
echo 3. Back
echo ===========================
set /p file_choice=Choose action 1-3:
if "%file_choice%"=="1" goto totalvram_1
if "%file_choice%"=="2" goto totalvram_2
if "%file_choice%"=="3" goto ram_opt
echo Wrong choice. please, try again.
pause
goto highram_lowvram

:totalvram_1
cls
echo ============================================
echo  Total VRAM - 1GB by default profile choice
echo ============================================
echo 1. Total VRAM - 1GB FLUX Async+CPU
echo 2. Total VRAM - 1GB FLUX Queue+CPU
echo 3. Back
echo ===========================
set /p file_choice=Choose action 1-3:
if "%file_choice%"=="1" goto async_cpu1
if "%file_choice%"=="2" goto queue_cpu1
if "%file_choice%"=="3" goto highram_lowvram
echo Wrong choice. please, try again.
pause
goto totalvram_1

:totalvram_2
cls
echo ============================================
echo  Total VRAM - 2GB by default profile choice
echo ============================================
echo 1. Total VRAM - 2GB FLUX Async+CPU
echo 2. Total VRAM - 2GB FLUX Queue+CPU
echo 3. Back
echo ===========================
set /p file_choice=Choose action 1-3:
if "%file_choice%"=="1" goto async_cpu2
if "%file_choice%"=="2" goto queue_cpu2
if "%file_choice%"=="3" goto highram_lowvram
echo Wrong choice. please, try again.
pause
goto totalvram_2

:async_cpu1
cls
set "file_name=main_entry.py"
set "source_folder=.\ram_opt\profiles_highram_lowvram\vram1\AsyncCPU"
set "destination_folder=.\modules_forge"
if exist "%source_folder%\%file_name%" (
    copy /y "%source_folder%\%file_name%" "%destination_folder%\%file_name%"
    echo Total VRAM - 1GB FLUX Async+CPU profile succesfully applied.
) else (
    echo Profile absent.
)
pause
goto totalvram_1

:queue_cpu1
cls
set "file_name=main_entry.py"
set "source_folder=.\ram_opt\profiles_highram_lowvram\vram1\QueueCPU"
set "destination_folder=.\modules_forge"
if exist "%source_folder%\%file_name%" (
    copy /y "%source_folder%\%file_name%" "%destination_folder%\%file_name%"
    echo Total VRAM - 1GB FLUX Queue+CPU profile succesfully applied.
) else (
    echo Profile absent.
)
pause
goto totalvram_1

:async_cpu2
cls
set "file_name=main_entry.py"
set "source_folder=.\ram_opt\profiles_highram_lowvram\vram2\AsyncCPU"
set "destination_folder=.\modules_forge"
if exist "%source_folder%\%file_name%" (
    copy /y "%source_folder%\%file_name%" "%destination_folder%\%file_name%"
    echo Total VRAM - 2GB FLUX Async+CPU profile succesfully applied.
) else (
    echo Profile absent.
)
pause
goto totalvram_2

:queue_cpu2
cls
set "file_name=main_entry.py"
set "source_folder=.\ram_opt\profiles_highram_lowvram\vram2\QueueCPU"
set "destination_folder=.\modules_forge"
if exist "%source_folder%\%file_name%" (
    copy /y "%source_folder%\%file_name%" "%destination_folder%\%file_name%"
    echo Total VRAM - 2GB FLUX Queue+CPU profile succesfully applied.
) else (
    echo Profile absent.
)
pause
goto totalvram_2

:memory_managmnt_prfl
cls
echo ===========================
echo     Memory Clear profile
echo ===========================
echo 1. Set Pure Forge Memory Management
echo 2. Set Clear Always Memory
echo 3. Back to main menu
echo ===========================
set /p file_choice=Choose action 1-3:
if "%file_choice%"=="1" goto pure_forge_clear
if "%file_choice%"=="2" goto always_cl
if "%file_choice%"=="3" goto main_menu
if not "%file_choice%"=="1" if not "%file_choice%"=="2" if not "%file_choice%"=="3" (
    echo Wrong choice. please, try again.
    pause
    goto memory_managmnt_prfl
)
pause
goto memory_managmnt_prfl

:pure_forge_clear
cls
set "file_name=memory_management.py"
set "source_folder=.\ram_opt\memory_def_forge"
set "destination_folder=.\backend"

if exist "%source_folder%\%file_name%" (
    copy /y "%source_folder%\%file_name%" "%destination_folder%\%file_name%"
    echo Pure Forge Memory Management succesfully applied.
) else (
    echo Profile absent.
)
pause
goto memory_managmnt_prfl

:always_cl
cls
set "file_name=memory_management.py"
set "source_folder=.\ram_opt\memory_always_clear_all"
set "destination_folder=.\backend"

if exist "%source_folder%\%file_name%" (
    copy /y "%source_folder%\%file_name%" "%destination_folder%\%file_name%"
    echo Clear Always Memory succesfully applied.
) else (
    echo Profile absent.
)
pause
goto memory_managmnt_prfl

:start_forge
cls
echo Starting engine...

set "HF_HOME=%cd%\.huggingface"
set "HF_HOME=%cd%_cache\huggingface"
set "XDG_CACHE_HOME=%cd%_cache"
set "HF_DATASETS_CACHE=%cd%_cache\huggingface\datasets"
set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--skip-python-version-check ^
--skip-version-check ^
--skip-torch-cuda-test ^
--xformers ^
--cuda-stream ^
--cuda-malloc ^
--no-half-vae ^
--precision half ^
--no-hashing ^
--upcast-sampling ^
--disable-nan-check

@REM UNCOMMENT FOLOWWING CODE (REPLACE COMMANDLINE_ARGS LINE WITH UNCOMMENTED CODE)AND SETUP EXTERNAL PATH FOR MODELS IF YOU HAVE THEM NOT INTERNALLY.
@REM set COMMANDLINE_ARGS=--skip-python-version-check ^
@REM --skip-version-check ^
@REM --skip-torch-cuda-test ^
@REM --xformers ^
@REM --cuda-stream ^
@REM --cuda-malloc ^
@REM --no-half-vae ^
@REM --precision half ^
@REM --no-hashing ^
@REM --upcast-sampling ^
@REM --disable-nan-check
@REM --ckpt-dir D:/COMFY_UI/ComfyUI/models/checkpoints ^
@REM --lora-dir D:/COMFY_UI/ComfyUI/models/loras ^
@REM --vae-dir D:/COMFY_UI/ComfyUI/models/vae ^
@REM --text-encoder-dir D:/COMFY_UI/ComfyUI/models/text_encoders ^
@REM --embeddings-dir D:/COMFY_UI/ComfyUI/models/embeddings ^
@REM --hypernetwork-dir D:/COMFY_UI/ComfyUI/models/hypernetworks ^
@REM --controlnet-dir D:/COMFY_UI/ComfyUI/models/controlnet

set EXPORT COMMANDLINE_ARGS=

@REM Uncomment following code to reference an existing A1111 checkout.
@REM set A1111_HOME=Your A1111 checkout dir
@REM
@REM set VENV_DIR=%A1111_HOME%/venv
@REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% ^
@REM  --ckpt-dir %A1111_HOME%/models/Stable-diffusion ^
@REM  --hypernetwork-dir %A1111_HOME%/models/hypernetworks ^
@REM  --embeddings-dir %A1111_HOME%/embeddings ^
@REM  --lora-dir %A1111_HOME%/models/Lora
@REM ADDITIONAL KEYS:
@REM  --always-offload-from-vram ^
@REM  (This flag will make things slower but less risky)
@REM (SPEED RELATED)
@REM --cuda-malloc ^ 
@REM (This flag will make things faster but more risky)
@REM --cuda-stream ^
@REM (This flag will make things faster but more risky)
@REM --pin-shared-memory ^ 
@REM (This flag will make things faster but more risky). Effective only when used together with --cuda-stream.
@REM --use-sage-attention ^
@REM     Uses SAGE attention implementation, from https://github.com/thu-ml/SageAttention. You need to install the library separately, as it needs triton.
@REM --attention-split ^
@REM     Use the split cross attention optimization. Ignored when xformers is used.
@REM --attention-quad ^
@REM     Use the sub-quadratic cross attention optimization . Ignored when xformers is used.
@REM --attention-pytorch ^
@REM     Use the new pytorch 2.0 cross attention function.
@REM --disable-attention-upcast ^
@REM    Disable all upcasting of attention. Should be unnecessary except for debugging.
@REM --force-channels-last ^
@REM     Force channels last format when inferencing the models.
@REM --disable-cuda-malloc ^
@REM     Disable cudaMallocAsync.
@REM --gpu-device-id ^
@REM     Set the id of the cuda device this instance will use.
@REM --force-upcast-attention ^
@REM     Force enable attention upcasting.
@REM 
@REM (VRAM related)
@REM --always-gpu ^
@REM     Store and run everything (text encoders/CLIP models, etc... on the GPU).
@REM --always-high-vram ^
@REM     By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.
@REM --always-normal-vram ^
@REM     Used to force normal vram use if lowvram gets automatically enabled.
@REM --always-low-vram ^
@REM     Split the unet in parts to use less vram.
@REM --always-no-vram ^
@REM    When lowvram isn't enough.
@REM --always-cpu ^
@REM     To use the CPU for everything (slow).
@REM 
@REM (float point type)
@REM --all-in-fp32 ^
@REM --all-in-fp16 ^
@REM --unet-in-bf16 ^
@REM --unet-in-fp16 ^
@REM --unet-in-fp8-e4m3fn ^
@REM --unet-in-fp8-e5m2 ^
@REM --vae-in-fp16 ^
@REM --vae-in-fp32 ^
@REM --vae-in-bf16 ^
@REM --clip-in-fp8-e4m3fn ^
@REM --clip-in-fp8-e5m2 ^
@REM --clip-in-fp16 ^
@REM --clip-in-fp32 ^
call webui.bat
