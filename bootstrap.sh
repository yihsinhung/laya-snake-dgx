#!/usr/bin/env bash
# =============================================================================
# laya-snake-dgx  BOOTSTRAP  —  一鍵安裝/修復(無腦重跑,冪等)
# 可在全新 DGX Spark GB10 上執行,也可在已裝好的機器上重跑修復。
#
#   bash ~/laya-snake-dgx/bootstrap.sh            # 完整安裝 + 驗證
#   bash ~/laya-snake-dgx/bootstrap.sh --gate     # 只做 CUDA 驗證
#
# 本機無 sudo,所以全部走原生 Python venv(不用 Docker)。
# =============================================================================
set -euo pipefail

VENV="${VENV:-$HOME/laya-snake-dgx-work/venv}"
WORK="${WORK:-$HOME/laya-snake-dgx-work}"
PKG="${PKG:-$HOME/laya-snake-dgx}"
MODEL="${MODEL:-convaiinnovations/laya-multilingual}"
TORCH_VER="${TORCH_VER:-2.14.0}"
HF_HOME="$WORK/hf"

log()  { printf '\n\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  [OK] %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
VP="$VENV/bin"

# --- 0. 前置 -------------------------------------------------------------
log "0 · prerequisites"
command -v python3 >/dev/null || fail "python3 not found"
command -v git >/dev/null || fail "git not found"
python3 -c "import venv" 2>/dev/null || fail "venv module missing (install python3-venv)"
ok "python3 + venv + git present"

# --- 1. venv -------------------------------------------------------------
log "1 · Python venv"
mkdir -p "$WORK"
if [ ! -x "$VP/python" ]; then
  python3 -m venv "$VENV"; ok "venv created at $VENV"
else
  ok "venv exists"
fi

# helper: already-installed check
has_pkg() { "$VP/python" -c "import $1" >/dev/null 2>&1; }

# --- 2. torch cu130 (aarch64) --------------------------------------------
log "2 · PyTorch ${TORCH_VER}+cu130 (aarch64) — first run downloads ~4GB"
if "$VP/python" -c "import torch; v=torch.__version__; assert 'cu130' in v" >/dev/null 2>&1; then
  ok "torch $( "$VP/python" -c 'import torch;print(torch.__version__)')"
else
  nohup "$VP/pip" install --no-cache-dir "torch==${TORCH_VER}" \
     --index-url https://download.pytorch.org/whl/cu130 >"$WORK/torch_install.log" 2>&1 &
  echo "  installing torch in background (pid $!)..."; wait
  tail -2 "$WORK/torch_install.log"
  "$VP/python" -c "import torch;assert 'cu130' in torch.__version__" || fail "torch install failed"
  ok "torch installed"
fi
"$VP/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
  || fail "torch.cuda.is_available() False — check NVIDIA driver"
ok "CUDA visible to torch"

# --- 3. laya runtime + snake deps ----------------------------------------
log "3 · laya runtime + snake deps (rich / Pillow)"
if has_pkg laya && has_pkg rich; then
  ok "laya + rich present"
else
  nohup "$VP/pip" install --no-cache-dir \
    "git+https://github.com/NandhaKishorM/laya.git" \
    "rich>=15,<16" "Pillow>=12,<13" >"$WORK/pkgs_install.log" 2>&1 &
  echo "  installing laya+deps in background (pid $!)..."; wait
  tail -2 "$WORK/pkgs_install.log"
  has_pkg laya || fail "laya install failed"
  ok "laya installed"
fi

# --- 4. snake package -----------------------------------------------------
log "4 · snake package (registers laya-snake)"
if "$VP/python" -c "import importlib.metadata as m; m.version('laya-snake-dgx')" >/dev/null 2>&1; then
  ok "laya-snake-dgx installed"
else
  "$VP/pip" install --no-cache-dir -e "$PKG" 2>&1 | tail -2
  ok "snake package installed"
fi

# --- 5. model weights -----------------------------------------------------
log "5 · model weights ($MODEL) -> $HF_HOME"
if [ -d "$HF_HOME/hub/models--${MODEL//\//--}" ]; then
  ok "weights present"
else
  HF_HOME="$HF_HOME" "$VP/python" -c \
    "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
  ok "weights downloaded"
fi

# --- 6. launcher ----------------------------------------------------------
log "6 · launcher run_snake.sh"
LAUNCH="$PKG/run_snake.sh"
cat >"$LAUNCH" <<EOF
#!/usr/bin/env bash
# Launcher for laya-snake on DGX Spark GB10 (created by bootstrap.sh)
export TORCH_DISABLE_NATIVE_JIT=1
export HF_HOME="$HF_HOME"
exec "$VP/laya-snake" "\$@"
EOF
chmod +x "$LAUNCH"
ok "launcher ready at $LAUNCH"

# --- 7. CUDA gate ---------------------------------------------------------
log "7 · CUDA gate test"
GATE=0
if [ "${1:-}" = "--gate" ]; then
  "$VP/python" "$PKG/gate_check.py" && GATE=$? || GATE=$?
else
  "$VP/python" "$PKG/gate_check.py" && GATE=$? || GATE=$?
fi
[ "$GATE" -eq 0 ] && ok "CUDA gate PASSED" || { echo "gate failed"; exit 1; }

echo
log "ALL DONE. Snapshot now:"
echo "  venv     : $VENV"
echo "  python   : $("$VP/python" --version)"
echo "  torch    : $("$VP/python" -c 'import torch;print(torch.__version__+" cuda="+str(torch.cuda.is_available()))')"
echo "  laya     : $("$VP/python" -c 'import laya;print(laya.__version__ if hasattr(laya,"__version__") else "0.3.22")')"
echo "  launcher : $LAUNCH"
echo "  run      : $LAUNCH --max-speed        (全速, 互動)"
echo "  run      : $LAUNCH --headless --steps 600 --max-speed   (無頭)"
