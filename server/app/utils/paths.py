from pathlib import Path

# 以本檔案位置為基準，避免受工作目錄 (cwd) 影響
_here    = Path(__file__).resolve()      # .../smartcare-server/app/utils/paths.py
SC_UTILS = _here.parent                  # .../app/utils
SC_APP   = SC_UTILS.parent               # .../app
SC_ROOT  = SC_APP.parent                 # .../smartcare-server

SC_MODELS = SC_ROOT / "models"
SC_ROUTES = SC_APP / "routes"

__all__ = ["SC_ROOT", "SC_APP", "SC_UTILS", "SC_MODELS", "SC_ROUTES"]
