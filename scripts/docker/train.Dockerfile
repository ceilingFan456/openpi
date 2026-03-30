# Dockerfile for pi0.5 training.
# Based on the serve_policy.Dockerfile with training-specific additions.
#
# Build:
#   docker build . -t openpi_train -f scripts/docker/train.Dockerfile
#
# Run (interactive):
#   docker run --rm -it --network=host --gpus=all --shm-size=16g \
#     -v .:/app \
#     -v ${OPENPI_DATA_HOME:-~/.cache/openpi}:/openpi_assets \
#     -e OPENPI_DATA_HOME=/openpi_assets \
#     openpi_train /bin/bash
#
# Run training directly:
#   docker run --rm --network=host --gpus=all --shm-size=16g \
#     -v .:/app \
#     -v ${OPENPI_DATA_HOME:-~/.cache/openpi}:/openpi_assets \
#     -e OPENPI_DATA_HOME=/openpi_assets \
#     openpi_train \
#     uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies (git-lfs needed by LeRobot, build tools for native extensions).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git git-lfs linux-headers-generic build-essential clang \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy from the cache instead of linking since it's a mounted volume.
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install Python and dependencies using the lockfile.
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Apply transformers_replace patches.
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /tmp/transformers_replace/* {} \
    && rm -rf /tmp/transformers_replace

# Default: allow JAX to use 90% of GPU memory for training.
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Default command opens a shell; override with your training command.
CMD ["/bin/bash"]
