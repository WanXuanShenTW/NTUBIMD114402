# app/utils/time_utils.py
from datetime import datetime

def now_str() -> str:
    # 伺服器本機時間，格式：YYYY-MM-DD HH:MM:SS
    return datetime.now().replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')

def format_time(self, ts_ms: int) -> str:
        """將時間戳記格式化為 YYYY-MM-DD HH:MM:SS"""
        return datetime.fromtimestamp(ts_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')