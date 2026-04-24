#!/bin/bash

# 1. Define paths
# Using a hash of the current directory to keep venvs unique on /mnt
REPO_HASH=$(pwd | md5sum | cut -c1-8)
REAL_VENV_PATH="/mnt/tmp/uv_envs/${REPO_HASH}/.venv"
REAL_CACHE_PATH="/mnt/tmp/uv_cache"

echo "🚀 Starting environment setup on /mnt..."

# 2. Create target directories on ephemeral storage
mkdir -p "$REAL_VENV_PATH"
mkdir -p "$REAL_CACHE_PATH"

# 3. Handle the symlink
if [ -L ".venv" ]; then
    echo "🔗 Symlink already exists."
elif [ -d ".venv" ]; then
    echo "⚠️ Warning: A physical .venv directory already exists. Moving it to /mnt/tmp..."
    mv .venv/* "$REAL_VENV_PATH/"
    rm -rf .venv
    ln -s "$REAL_VENV_PATH" .venv
else
    echo "🔗 Creating new symlink to /mnt..."
    ln -s "$REAL_VENV_PATH" .venv
fi

# 4. Configure uv to use the ephemeral cache
export UV_CACHE_DIR="$REAL_CACHE_PATH"

# 5. Sync the environment
echo "📦 Running uv sync..."
uv sync

# 6. Output the Python Interpreter Path
echo "-----------------------------------------------"
echo "✅ Setup Complete!"
echo "Python Interpreter Path:"
echo "$(pwd)/.venv/bin/python"
echo "-----------------------------------------------"