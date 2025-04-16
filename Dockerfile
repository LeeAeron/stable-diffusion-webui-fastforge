# Use an official lightweight Python base image
FROM python:3.10-slim

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
COPY requirements_versions.txt .

# Install Python dependencies and specify extra index for PyTorch
RUN git clone https://github.com/salesforce/BLIP.git /app/repositories/BLIP
RUN git clone https://github.com/lllyasviel/huggingface_guess.git /app/repositories/huggingface_guess
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git /app/repositories/stable-diffusion-webui-assets
RUN git clone https://github.com/lllyasviel/google_blockly_prototypes.git /app/repositories/google_blockly_prototypes
RUN pip install --no-cache-dir -r requirements_versions.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the rest of the application files
COPY . .

# Set environment variables to streamline application performance
ENV PYTHONUNBUFFERED=1

# Expose the port the application will run on
EXPOSE 7860

# Set the default command to run the application
CMD ["python", "webui.py", "--skip-python-version-check" , "--skip-version-check", "--skip-torch-cuda-test", "--xformers", "--cuda-stream", "--cuda-malloc", "--no-half-vae", "--no-hashing", "--upcast-sampling", "--disable-nan-check", "--listen", "--port=7860"]
