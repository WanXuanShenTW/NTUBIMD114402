# app/utils/time_utils.py
from datetime import datetime
import pytz

def now_str() -> str:
    taipei_tz = pytz.timezone("Asia/Taipei")
    return datetime.now(taipei_tz).replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')

def format_time(self, ts_ms: int) -> str:
    taipei_tz = pytz.timezone("Asia/Taipei")
    return datetime.fromtimestamp(ts_ms / 1000, taipei_tz).strftime('%Y-%m-%d %H:%M:%S')