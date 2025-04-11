# sd-forge-fum
Replacement for the BuildIn FreeU Implementation in Forge with additional options for animations

![fum1](https://github.com/user-attachments/assets/a9961ff0-55a2-405b-bc4e-5ea151bf5292)

This is a try out project to check how to animate a picture using a single untouched seed. 
The idea was to alter the values of the S1 and / or S2 sliders while the seed is unchanged.
After many tests and tips by the forge community members, this is the first release version.
The code is mostly the original code implemented in the forge webui by lllyasviel and the
many contributors, the additional options for the randomised or linear parameter moves are
what this project is all about written by me so far..

Make sure to install the script to your forge webui using the link https://github.com/zeittresor/sd-forge-fum.git
(It seems to be useful to disable the build-in extension "FreeU" in Forge WebUI to have the settings not a second time).

The Parameters in FUM (FreeU-Move) are mostly the same like in the original implementation but with the little
difference that you now have some more options you can enable using checkboxes.

![fum0](https://github.com/user-attachments/assets/646f1d48-f1b2-486c-b2e1-af356d535950)

How to use:
Make sure u generate a single Image first, if you like the generated image, make sure you do NOT USE any wildcard
in your prompt field or have dynamic prompting enabled - the reason for this is that we do not want anything what
changes instead of the S1 / S2 values in the generation procedure.
Make sure you set the seed of the image now to the fixed value of the Image you are happy with.
Enable the checkbox in the FreeU-Move (FUM) extension for the option you want to try.
Click with the right mousekey onto the "Generate" Button and select in the context menu "Generate forever".
Now let the process generate some images.. after you have generated for ex. 30 images stop the procedure.
Go to your output folder - you should see now all the images - mostly looking same, but with slightly changes.
Put the script you can find in the extras folder into it and use it to generate the smooth transition images
for each image in transition to the next .. the option to automaticly make a video from all the generated images
is not working until now but you can use other free scripts to add "the frames" to a movie. btw. there are
some commandline options for the makevid.py script like automatic using lanczos upscaling you can use on low end
machines to get good results.

The Idea was born in the discussion group of the forge webui https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/1871
Thanks to all people helping to make this finaly work 😍

btw. you do not need a "NASA Computer" or a modern graphic card to produce a single image animation in top resolution with it just some more time 😉

.. an additional idea i had is to use it with the layer diffusion extension in combination to to make the sometimes "unstable" background "invisible". Using
that methode it "should be" possible to resurrect a person of a photo to a somehow realistic 3D looking person with a additional background you can add to the
"now transparent" alpha channel - but thats only a idea for a future release - i have not tested this so far 😉

Workflow for Animation / Video generating:
Make sure you did the following:

- You have a Prompt without any wildcard like "_ [wildcardname] _" or {value1|value2|..} in the text
- You have set the seed to a fixed value
- You have disabled the dynamic prompt extension if you normaly use it (checked by default) just uncheck this now
- Disable the FreeU extension.
- Enable the FUM (FreeU-Move) extension.
- Check "Random UNet Move", "Simple UNet S1 Move" or both.
- Set the batch count to 1
- Make sure your Output folder is empty (if you want to monitor the progress of the video building progress)*
- Click with the right mousebutton on the generate button and select "Generate forever".

* You can use my script "timedviewer" (https://github.com/zeittresor/timedviewer) to monitor the video building progress in realtime, i have added a preset for FUM there.

* Tip #1: While the generation process is active you can still make little changes to the prompt field that modifies the image sometimes for example "view from left front." changed to "view from left side."
* Tip #2: To get a more stable animation without alot flickering its helping to add something like "..in front of a black background, on top of a black floor".
  
  
Source: https://github.com/zeittresor/sd-forge-fum
