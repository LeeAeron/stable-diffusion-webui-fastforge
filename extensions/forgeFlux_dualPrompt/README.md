## extra features extension for Forge2 webUI ##

install:
**Extensions** tab, **Install from URL**, use URL for this repo

Necessary for Flux Redux - update **requirements_versions.txt** in the webUI directory.
```
diffusers>=0.32.0
```

usage:
1. Enable the extension
2. use the Prompt textbox as normal
3. prompt for first text encoder
4. enter word **SPLIT**, use surrounding whitespace/newline as you like
5. prompt for second text encoder
6. Generate

Flux (CLIP-L / T5) assumed best usage is tags for first prompt, natural language for second.
SDXL (CLIP-L / CLIP-G) would typically be tags for both but depends on training: seems likely that same captions would be used for both CLIPs so use may be limited.

update 1:
* added sdXL, why not? In process, prepped code for possible future additions of SD3 / Hunyuan / others?.
* changed unpatch location. Previously unpatched ASAP therefore wouldn't have applied during hires fix (only relevant if hires fix had new prompt which included SPLIT). In practice, unlikely to be relevant at all.
* only patches if using appropriate models (Flux, sdXL, *update* SD3). Previously always patched if extension enabled, which would have left the patch in place if using a model which didn't use the patched function which would have unpatched itself but couldn't.
* Force clearance of cached conds when extension enabled/disabled.
* for explanation of these egregious errors, see line 2 of this document.

update 2:
* added control for Shift parameter (affects calculation of sigmas used to guide denoising per step). This is for the *Simple* scheduler only.
* Dynamic Shift is a different method of calculating the way Shift affects sigmas. From brief testing, it seems to work very badly with higher Shift values, but may be better at low values. It's the way Forge calculates, so included for completeness / reproducibility. Dynamic Shift is the term diffusers uses.

update 3:
* changed to base/max shift implementation. Max Shift > 0.0 means using the dynamic method.
* added separate controls for HighRes fix. Leave at 0 to use same values as non-HR.
* added automatic backup/restore of original sigmas, as this overwrites (so previously disabling extension left the last used settings in place)

update 4:
* added control over model prediction type. Particularly (only?) relevant for v-prediction models. Renamed to 'Forge2 extras'.

update 5:
* added option to disable individual text encoders. Different results to sending an empty prompt.

update 6:
* control of device used for text encoding
* SD3 text encoder control

update 7:
* simple FluxTools support (canny and depth). Control image will be automatically resized to UI width and height. Distilled guidance 10+ seems best.

update 7.5:
* add FluxTools Redux. Necessary models will be downloaded on first use, just under 1GB. Redux requires `diffusers>=0.32.0`.

update 8:
* add FluxTools Fill. Fill is prioritised, so if an image is in the Fill tab then the Fill process will be used. If not, an image in the Canny / Depth tab will cause that process to be used. Redux can be used in combination with C/D/F, or alone with a standard Flux model. You may need to lower your `GPU Weights` setting by ~250MB. If you have the wrong model selected in Forge, you'll get an error along the lines of `mat1 and mat2 shapes cannot be multiplied`.
