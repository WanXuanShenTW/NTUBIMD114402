from typing import Optional, Dict, Any
from aiomysql.cursors import DictCursor

async def upsert_prefs(conn, user_id: int, weekdays: str, hour: int, minute: int) -> int:
    """新增或更新通知偏好設定"""
    sql = """
        INSERT INTO notification_preferences (user_id, weekdays, hour, minute)
        VALUES (%s, %s, %s, %s) AS new_prefs
        ON DUPLICATE KEY UPDATE
            weekdays = new_prefs.weekdays,
            hour = new_prefs.hour,
            minute = new_prefs.minute
    """
    try:
        async with conn.cursor() as cursor:
            await cursor.execute(sql, (user_id, weekdays, hour, minute))
            await conn.commit()
            return cursor.rowcount
    except Exception as e:
        await conn.rollback()
        raise Exception(f"更新通知偏好失敗: {e}")

async def get_prefs_by_user_id(conn, user_id: int) -> Optional[Dict[str, Any]]:
    """根據用戶ID獲取通知偏好設定"""
    sql = "SELECT user_id, weekdays, hour, minute FROM notification_preferences WHERE user_id=%s"
    try:
        async with conn.cursor(DictCursor) as cursor:
            await cursor.execute(sql, (user_id,))
            return await cursor.fetchone()
    except Exception as e:
        raise Exception(f"查詢通知偏好失敗: {e}")