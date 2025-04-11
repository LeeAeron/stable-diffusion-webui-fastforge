# MakeVid Script (beta 0.1)
# Source: https://github.com/zeittresor/sd-forge-fum

import cv2 # pip install opencv-python
import numpy as np 
import os
import subprocess
from PIL import Image #pip install pillow
import argparse

# requires FFMPEG in the same folder like the script to not just create intermediate images but also a vid
# if no ffmpeg is found it would create the intermediate images but result with a error after that process.
# https://www.ffmpeg.org/download.html


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def generate_intermediate_images(img1, img2, num_intermediates=3):
    intermediates = []
    for i in range(1, num_intermediates + 1):
        alpha = i / (num_intermediates + 1)
        intermediate = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        intermediates.append(intermediate)
    return intermediates

def save_images(images, base_filename, output_folder):
    base_name, ext = os.path.splitext(base_filename)
    for idx, img in enumerate(images):
        output_filename = f"{base_name}_{idx + 1}{ext}"
        cv2.imwrite(os.path.join(output_folder, output_filename), img)

def create_video_from_images(folder, output_video, fps=30):
    subprocess.run([
        'ffmpeg', '-framerate', str(fps), '-i', os.path.join(folder, '%*.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video
    ])

def upscale_image(image, scale_factor=2):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    new_size = (pil_image.width * scale_factor, pil_image.height * scale_factor)
    upscaled_image = pil_image.resize(new_size, Image.LANCZOS)
    return cv2.cvtColor(np.array(upscaled_image), cv2.COLOR_RGB2BGR)

def resize_images(images, size):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, size)
        resized_images.append(resized_img)
    return resized_images

def ensure_three_channels(images):
    converted_images = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        converted_images.append(img)
    return converted_images

def main(folder, num_intermediates=3, output_video='output.mp4', upscale=False):
    images, filenames = load_images_from_folder(folder)
    if upscale:
        images = [upscale_image(img) for img in images]
    
    images = ensure_three_channels(images)
    
    if images:
        size = (images[0].shape[1], images[0].shape[0])
        images = resize_images(images, size)
    
    for i in range(len(images) - 1):
        intermediates = generate_intermediate_images(images[i], images[i + 1], num_intermediates)
        save_images(intermediates, filenames[i], folder)
    create_video_from_images(folder, output_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Vid from images for unet move animation by Zeittresor')
    parser.add_argument('folder', type=str, help='Folder containing images ex. python createvid.py . to use the current folder')
    parser.add_argument('--num_intermediates', type=int, default=3, help='Number of intermediate images to generate (default: 3)')
    parser.add_argument('--output_video', type=str, default='output.mp4', help='Output video file name (default: output.mp4)')
    parser.add_argument('--upscale', action='store_true', help='Upscale images by 2x using lanczos before processing (set this to scale images up)')
    args = parser.parse_args()
    main(args.folder, args.num_intermediates, args.output_video, args.upscale)
