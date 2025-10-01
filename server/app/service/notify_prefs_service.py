from typing import List, Tuple, Optional, Dict, Any
from ..db import Database
from ..dao.notify_prefs_dao import upsert_prefs, get_prefs_by_user_id
from ..dao.line_binding_dao import get_user_by_line_user_id

WEEK_NAMES = ["日","一","二","三","四","五","六"]

def parse_sel(sel: str) -> List[int]:
    if not sel: return []
    return [int(x) for x in sel.split(",") if x != ""]

def toggle_day(sel: List[int], d: int) -> List[int]:
    s = set(sel)
    if d in s: s.remove(d)
    else: s.add(d)
    return sorted(s)

def human_days(days: List[int]) -> str:
    if not days: return "（未選）"
    return "、".join(f"週{WEEK_NAMES[d]}" for d in days)

def human_time(h: int, m: int) -> str:
    return f"{h:02d}:{m:02d}"

async def save_prefs_by_line_uid(line_user_id: str, days: List[int], hour: int, minute: int) -> Tuple[bool, str]:
    """保存通知偏好設定"""
    try:
        async with Database.connection() as conn:
            user = await get_user_by_line_user_id(conn, line_user_id)
            if not user:
                return False, "尚未綁定帳號，請先完成綁定。"
            
            weekdays = ",".join(map(str, days))
            # 移除 timezone 參數
            await upsert_prefs(conn, user["user_id"], weekdays, hour, minute)
            
            return True, f"設定完成 ✅\n你將在 {human_days(days)} 的 {human_time(hour, minute)} 收到通知。"
            
    except Exception as e:
        print(f"保存通知偏好時發生錯誤: {e}")
        return False, f"設定失敗：{str(e)}"

async def get_prefs_for_line_uid(line_user_id: str) -> Optional[Dict[str, Any]]:
    """獲取 LINE 用戶的通知偏好設定"""
    try:
        async with Database.connection() as conn:
            user = await get_user_by_line_user_id(conn, line_user_id)
            if not user:
                return None
            
            prefs = await get_prefs_by_user_id(conn, user["user_id"])
            if not prefs:
                return None
            
            return {
                "user_id": prefs["user_id"],
                "weekdays": [int(x) for x in prefs["weekdays"].split(",") if x],
                "hour": prefs["hour"],
                "minute": prefs["minute"],
            }
            
    except Exception as e:
        print(f"獲取通知偏好失敗：{e}")
        return None