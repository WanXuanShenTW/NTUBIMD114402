#!/bin/bash
set -euo pipefail

# 可選：這裡先覆寫你之前過敏的取得連線逾時（建議先關掉）
export USE_POOL_TIMEOUT=false
# 若未來要保留超時機制，改為：
# export USE_POOL_TIMEOUT=true
# export POOL_TIMEOUT=60

# 若你用 venv，且這支腳本不是在已啟動的 venv 下執行，可打開下面兩行
# VENV_DIR="$(dirname "$0")/.venv"
# source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1

# 關鍵設定：
# 1) --workers 1        → 避免每個 worker 都建立一個 DB 連線池（疊爆）
# 2) --limit-concurrency 16 → 控制同時處理請求數，避免把 pool (max=20) 用滿；保留緩衝
# 3) 其他：把存活時間與日誌設得清楚一點，方便追 Log
uvicorn run:app \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1 \
  --limit-concurrency 16 \
  --timeout-keep-alive 30 \
  --timeout-graceful-shutdown 5 \
  --log-level info \
  --access-log \
  --no-server-header \
  --proxy-headers
