# Start from the full-featured PyTorch image. This guarantees that all
# system libraries are compatible with the installed Python environment.
FROM nvcr.io/nvidia/pytorch:25.05-py3

WORKDIR /app

# Copy the full project context. This is necessary for the next step
# to correctly install dependencies from your pyproject.toml file.
# Using a .dockerignore file is highly recommended to keep this small.
COPY . .

# RUN a single, multi-line command to perform all setup and cleanup.
# This creates only one layer, making the image smaller.
RUN apt-get update && \
    # Install build tools
    apt-get install -y --no-install-recommends python3-pip && \
    pip install --no-cache-dir uv && \
    \
    # Create the virtual environment
    uv venv .venv && \
    . .venv/bin/activate && \
    # Explicitly install pip and setuptools into the new venv
    uv pip install --no-cache-dir pip setuptools && \
    # Now install the project's dependencies using pyproject.toml
    # This installs dependencies without installing the project itself in editable mode.
    uv pip install --no-cache-dir . && \
    \
    # --- Aggressive Cleanup ---
    # Remove all package manager caches to shrink the image
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    uv cache clean && \
    rm -rf /root/.cache && \
    # Remove the copied source code to keep the final image as a runtime-only environment
    rm -rf /app/*

# The image is now a self-contained runtime environment with an empty /app directory,
# ready for your code to be mounted.