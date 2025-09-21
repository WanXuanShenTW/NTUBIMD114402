# app/utils/time_utils.py
from datetime import datetime

def now_str() -> str:
    # 伺服器本機時間，格式：YYYY-MM-DD HH:MM:SS
    return datetime.now().replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
