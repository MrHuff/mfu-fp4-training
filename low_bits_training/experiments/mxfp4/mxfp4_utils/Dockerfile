# Start from the same base image you are already using
FROM nvcr.io/nvidia/pytorch:25.05-py3

# Set a working directory inside the image. This is where your code will live.
WORKDIR /app

# Copy your project's source code into the image's working directory
# This assumes your Dockerfile is in the root of your 'low-bits-training' project
COPY . .

# RUN the setup commands ONCE during the image build.
# This creates the .venv inside the image itself.
RUN apt-get update && apt-get install -y python3-pip && \
    pip install uv && \
    uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# The image is now built. The default command can be set here, but we will
# override it in the Kubernetes YAML for more flexibility.
