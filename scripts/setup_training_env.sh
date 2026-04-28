#!/usr/bin/env bash
# setup_training_env.sh — Install all dependencies for pi0.5 training on a fresh machine.
#
# Designed for batch jobs (e.g., SLURM, Azure ML, cloud VMs) where you start from
# a base image with NVIDIA drivers + CUDA already available.
#
# Usage:
#   bash scripts/setup_training_env.sh              # install deps only
#   bash scripts/setup_training_env.sh --run-train "pi05_libero --exp-name=my_exp --overwrite"
#
# Prerequisites:
#   - Ubuntu 22.04 (or compatible)
#   - NVIDIA GPU drivers installed
#   - CUDA 12.x runtime available
#   - Internet access (for downloading packages + model checkpoints)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- Parse arguments ----------
TRAIN_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-train)
            TRAIN_ARGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo "=========================================="
echo " openpi training environment setup"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# ---------- 1. System packages ----------
echo "[1/6] Installing system packages..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        git git-lfs build-essential clang curl ca-certificates \
        linux-headers-generic libgl1-mesa-glx libglib2.0-0
elif command -v yum &>/dev/null; then
    sudo yum install -y git git-lfs gcc gcc-c++ clang curl ca-certificates mesa-libGL glib2
else
    echo "Warning: Unknown package manager. Ensure git, git-lfs, build-essential, clang are installed."
fi

git lfs install 2>/dev/null || true

# ---------- 2. Install uv ----------
echo "[2/6] Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# ---------- 3. Initialize git submodules ----------
echo "[3/6] Initializing git submodules..."
git submodule update --init --recursive 2>/dev/null || true

# ---------- 4. Create venv & install Python deps ----------
echo "[4/6] Installing Python dependencies (this may take a while)..."
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# ---------- 5. Apply transformers patches ----------
echo "[5/6] Applying transformers_replace patches..."
TRANSFORMERS_DIR=$(uv run python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
if [ -d "src/openpi/models_pytorch/transformers_replace" ]; then
    cp -r src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR/"
    echo "Patched transformers at: $TRANSFORMERS_DIR"
else
    echo "Warning: transformers_replace directory not found, skipping patch."
fi

# ---------- 6. Verify installation ----------
echo "[6/6] Verifying installation..."
uv run python -c "
import jax
import torch
import flax
import transformers

print(f'  JAX version:          {jax.__version__}')
print(f'  JAX devices:          {jax.devices()}')
print(f'  PyTorch version:      {torch.__version__}')
print(f'  CUDA available:       {torch.cuda.is_available()}')
print(f'  Flax version:         {flax.__version__}')
print(f'  Transformers version: {transformers.__version__}')
"
echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo ""
echo "To run training:"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<name> --overwrite"
echo ""
echo "To save checkpoints to shared storage (e.g., on Singularity):"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<name> --checkpoint-base-dir=/mnt/default_storage/qiming/openpi/checkpoints --overwrite"
echo ""
echo "Example:"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --checkpoint-base-dir=/mnt/default_storage/qiming/openpi/checkpoints --overwrite"
echo ""

# ---------- Optionally run training ----------
if [ -n "$TRAIN_ARGS" ]; then
    echo "Starting training with args: $TRAIN_ARGS"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    # shellcheck disable=SC2086
    exec uv run scripts/train.py $TRAIN_ARGS
fi
