#!/usr/bin/env bash
# =============================================================================
# laya-snake-dgx  AFTER-REBOOT  —  重開機後一鍵恢復執行蛇
#
#   bash ~/laya-snake-dgx/after_reboot.sh                # 驗證 + 開 FULL 互動窗
#   bash ~/laya-snake-dgx/after_reboot.sh --headless     # 開無頭 benchmark
#
# 重開機不會清掉磁碟上的 venv 與權重,所以這裡只是「驗證 + 重新 launch」。
# ─────────────────────────────────────────────────────────────────────────
# 安裝導向:全新機器或想重裝 → 用 bootstrap.sh
# 重開機後:一切已在        → 用 after_reboot.sh
# =============================================================================
set -euo pipefail

PKG="${PKG:-$HOME/laya-snake-dgx}"
LAUNCH="$PKG/run_snake.sh"
VP="$HOME/laya-snake-dgx-work/venv/bin"
DISPLAY="${DISPLAY:-:0}"
XAUTH="/run/user/1000/gdm/Xauthority"
MODE="${1:-}"

# --- 驗證環境 (都在磁碟,重開機不變) -------------------------------------
check() {
  [ -x "$VP/python" ] || { echo "❌ venv 不見 — 先跑 bootstrap.sh"; return 1; }
  echo "✅ python $("$VP/python" --version 2>&1)"
  "$VP/python" -c "import torch; assert torch.cuda.is_available(); print('✅ torch', torch.__version__, 'CUDA OK')"
  "$VP/python" -c "import laya; print('✅ laya', laya.__version__ if hasattr(laya,'__version__') else '0.3.22')"
  [ -x "$LAUNCH" ] || { echo "❌ launcher 不見 — 先跑 bootstrap.sh"; return 1; }
  [ -d "$HOME/laya-snake-dgx-work/hf/hub" ] && echo "✅ weights cached" || echo "⚠️ weights cache 找不到"
}
check || exit 1

# --- venv 提示 ---------------------------------------------------------------
echo
echo "🔧 若要手動用 python(不透過 launcher),先啟動 venv:"
echo "    source ~/laya-snake-dgx-work/venv/bin/activate"
echo "    python --version   # 確認已進 venv"
echo

# --- 確認 X 桌面可用 ------------------------------------------------------
export DISPLAY="$DISPLAY"
export XAUTHORITY="$XAUTH"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"
if xdpyinfo >/dev/null 2>&1; then
  echo "✅ display :$DISPLAY reachable"
else
  echo "⚠️ display :$DISPLAY 還不可用 — 如果 GNOME 桌面還沒起來,稍等或登入後再跑一次。"
fi

# --- launch ---------------------------------------------------------------
if [ "$MODE" = "--headless" ]; then
  nohup "$LAUNCH" --headless --steps 600 --max-speed \
    >"$HOME/laya-snake-dgx-work/after_reboot_headless.log" 2>&1 &
  echo "📦 headless 啟動 (pid $!) — 看 $HOME/laya-snake-dgx-work/after_reboot_headless.log"
else
  nohup gnome-terminal --geometry=104x35 --title="Laya Snake FULL" \
    -- bash -c '"$LAUNCH" --max-speed; exec bash' \
    >"$HOME/laya-snake-dgx-work/after_reboot.log" 2>&1 &
  echo "🖥️  FULL 互動窗已送出 (pid $!) — 請看 :$DISPLAY 畫面"
fi
echo "⏳ 幾秒後可用下面指令確認:"
echo "     pgrep -af laya-snake"
echo "     DISPLAY=:0 xwininfo -root -tree | grep 'Laya Snake'"
