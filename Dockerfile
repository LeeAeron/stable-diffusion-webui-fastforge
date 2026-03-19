# Use an official lightweight Python base image
FROM python:3.12.7-slim

# Set the working directory in the container
WORKDIR /app

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    libstdc++6 \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to leverage Docker's caching
COPY prerequirements-versions.txt .
COPY requirements_versions.txt .

# Install Python dependencies and specify extra index for PyTorch
RUN git clone https://github.com/salesforce/BLIP.git /app/repositories/BLIP
RUN git clone https://github.com/lllyasviel/huggingface_guess.git /app/repositories/huggingface_guess
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git /app/repositories/stable-diffusion-webui-assets
RUN git clone https://github.com/lllyasviel/google_blockly_prototypes.git /app/repositories/google_blockly_prototypes
RUN pip install --no-cache-dir -r prerequirements-versions.txt
RUN pip install --no-cache-dir -r requirements_versions.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Copy the rest of the application files
COPY . .

# Set environment variables to streamline application performance
ENV PYTHONUNBUFFERED=1

# Expose the port the application will run on
EXPOSE 7860

# Set the default command to run the application for GTX-RTX20xx-RTX40xx series cards
CMD ["python", "webui.py", "--cuda-stream", "--cuda-malloc", "--pin-shared-memory", "--xformers", "--xformers-flash-attention", "--disable-gpu-warning", "--opt-sdp-attention", "--precision half", "--fast-fp16", "--listen", "--port=7860"]
# Set the default command to run the application for RTX50xx series cards
#CMD ["python", "webui.py", "--cuda-stream", "--cuda-malloc", "--pin-shared-memory", "--disable-xformers", "--disable-gpu-warning", "--opt-sdp-attention", "--precision half", "--fast-fp16", "--listen", "--port=7860"]
