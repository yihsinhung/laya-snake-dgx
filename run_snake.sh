#!/usr/bin/env bash
# Launcher for laya-snake on DGX Spark GB10 (created by bootstrap.sh)
export TORCH_DISABLE_NATIVE_JIT=1
export HF_HOME="/home/asus/laya-snake-dgx-work/hf"
exec "/home/asus/laya-snake-dgx-work/venv/bin/laya-snake" "$@"
